import socket
import struct
import unittest
import threading
from unittest.mock import patch
import json
import hashlib
from prefill_relay.transport import Receiver, StreamSender, send_frame
from prefill_relay.cache import HybridCache

class TransportTests(unittest.TestCase):
    def test_parallel_streaming_ack_and_release(self):
        pool=HybridCache(block_bytes=16,capacity_blocks=256)
        with Receiver(max_bytes=4096) as receiver:
            with StreamSender(receiver.address,'request',connections=2,pool=pool) as sender:
                for i in range(4): sender.submit(str(i),bytes([i])*100)
            self.assertEqual(receiver.take('request'),{str(i):bytes([i])*100 for i in range(4)})
            self.assertEqual(sender.stats['bytes'],400)
            self.assertEqual(pool.used_blocks,0)
            self.assertEqual(receiver.buffered_bytes,0)

    def test_sustained_chunk_stream_reuses_two_connections(self):
        from prefill_relay.transport import assemble_chunks
        payload = bytes(range(32)) * 32
        barrier = threading.Barrier(2)
        opened = []

        def connect(address, timeout):
            sock = socket.create_connection(address, timeout=timeout)
            opened.append(sock)
            barrier.wait(timeout=2)
            return sock

        with Receiver() as receiver:
            with StreamSender(receiver.address, 'striped', connections=2, connect=connect) as sender:
                sender.submit_blob('layer', payload, chunk_bytes=32)
            self.assertEqual(assemble_chunks(receiver.take('striped')), {'layer': payload})
        self.assertEqual(len(opened), 2)
        self.assertTrue(all(sock.fileno() == -1 for sock in opened))
        self.assertEqual(sender.stats['frames_acked'], 32)
        self.assertEqual(sender.stats['connections_opened'], 2)

    def test_lost_ack_reconnects_without_double_allocating(self):
        pool = HybridCache(block_bytes=16, capacity_blocks=2)
        original_send = socket.socket.sendall
        lost = []
        def drop_first_ack(sock, data):
            if data == b'ACK' and not lost:
                lost.append(True)
                sock.shutdown(socket.SHUT_RDWR)
                raise ConnectionResetError('lost ACK')
            return original_send(sock, data)

        with Receiver(max_bytes=16) as receiver, patch.object(socket.socket, 'sendall', drop_first_ack):
            with StreamSender(receiver.address, 'retry', connections=1, pool=pool) as sender:
                sender.submit('layer', b'0123456789abcdef')
            self.assertEqual(receiver.buffered_bytes, 16)
            self.assertEqual(receiver.take('retry'), {'layer': b'0123456789abcdef'})
        self.assertEqual(pool.used_blocks, 0)
        self.assertEqual(sender.stats['connections_opened'], 2)
        self.assertEqual(sender.stats['retries'], 1)
        self.assertEqual(sender.stats['frames_acked'], 1)

    def test_persistent_socket_counts_retransmission_deltas(self):
        values = iter((1, 1, 3))
        def sample(sock):
            return dict(tcp_info_available=True, rtt_ms=1., retransmissions=next(values))

        with Receiver() as receiver, patch('prefill_relay.transport.tcp_metrics', sample):
            with StreamSender(receiver.address, 'sample', connections=1) as sender:
                for i in range(3):
                    sender.submit(str(i), b'x')
        self.assertEqual(sender.stats['connections_opened'], 1)
        self.assertEqual(sender.stats['retransmissions'], 3)

    def test_receiver_shutdown_wakes_idle_stream(self):
        import time
        receiver = Receiver(timeout=5)
        with receiver:
            sock = socket.create_connection(receiver.address)
            self.addCleanup(sock.close)
            payload = b'x'
            header = json.dumps(dict(request='idle', layer='0', size=1,
                                     sha256=hashlib.sha256(payload).hexdigest())).encode()
            sock.sendall(struct.pack('!I', len(header)) + header + payload)
            self.assertEqual(sock.recv(3), b'ACK')
            started = time.monotonic()
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(sock.recv(1), b'')

    def test_receiver_bound_and_idempotent_retry(self):
        with Receiver(max_bytes=4) as r:
            send_frame(r.address,'r','l',b'abcd')
            send_frame(r.address,'r','l',b'abcd')
            with self.assertRaises(RuntimeError): send_frame(r.address,'r','x',b'a')
            with self.assertRaises(RuntimeError): send_frame(r.address,'r','l',b'zzzz')
            self.assertEqual(r.take('r'),{'l':b'abcd'})

    def test_reservation_commits_before_ack(self):
        with Receiver(max_bytes=4) as r:
            real_send=socket.socket.sendall
            observed=[]
            def inspect_ack(sock,data,*args,**kwargs):
                if data==b'ACK': observed.append((r.buffered_bytes,r.pending_bytes))
                return real_send(sock,data,*args,**kwargs)
            with patch.object(socket.socket,'sendall',inspect_ack):
                send_frame(r.address,'r','l',b'abcd')
            self.assertEqual(observed,[(4,0)])

    def test_checksum_and_truncation_release_capacity(self):
        with Receiver(max_bytes=4,timeout=.2) as r:
            header=json.dumps(dict(request='r',layer='x',size=4,sha256=hashlib.sha256(b'abcd').hexdigest())).encode()
            with socket.create_connection(r.address,timeout=2) as sock:
                sock.sendall(struct.pack('!I',len(header))+header+b'xxxx')
                self.assertEqual(sock.recv(3),b'ERR')
            with socket.create_connection(r.address,timeout=2) as sock:
                sock.sendall(struct.pack('!I',len(header))+header+b'ab')
                sock.shutdown(socket.SHUT_WR)
                self.assertEqual(sock.recv(3),b'ERR')
            send_frame(r.address,'r','good',b'abcd')
            self.assertEqual(r.take('r'),{'good':b'abcd'})

    def test_partial_request_discard_reclaims_receiver_memory(self):
        with Receiver(max_bytes=4) as r:
            send_frame(r.address,'partial','first',b'abcd')
            with self.assertRaises(RuntimeError):
                with StreamSender(r.address,'partial') as sender:
                    sender.submit('second',b'x')
            self.assertEqual(r.buffered_bytes,4)
            r.discard('partial')
            r.discard('partial')
            self.assertEqual(r.buffered_bytes,0)
            send_frame(r.address,'next','layer',b'abcd')
            self.assertEqual(r.take('next'),{'layer':b'abcd'})

    def test_oversized_header_rejected(self):
        with Receiver() as r:
            with socket.create_connection(r.address,timeout=2) as sock:
                sock.sendall(struct.pack('!I',1000000))
                self.assertTrue(sock.recv(100).startswith(b'ERR'))

    def test_failed_send_releases_explicitly_aborted_transfer(self):
        pool=HybridCache(block_bytes=16,capacity_blocks=256)
        with Receiver(max_bytes=1) as r:
            with self.assertRaises(RuntimeError):
                with StreamSender(r.address,'r',pool=pool) as sender: sender.submit('l',b'too big')
        self.assertEqual(pool.used_blocks,0)

if __name__=='__main__': unittest.main()
