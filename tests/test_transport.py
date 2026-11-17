import socket
import struct
import unittest
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
