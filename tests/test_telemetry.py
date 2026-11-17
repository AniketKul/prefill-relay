import unittest
from prefill_relay.telemetry import LinkMonitor, RolePool
from prefill_relay.transport import Receiver, StreamSender, assemble_chunks

class TelemetryTests(unittest.TestCase):
    def test_abort_rejects_inflight_late_commit(self):
        import socket, struct, json, hashlib, time
        from prefill_relay.transport import Receiver
        payload=b'late payload'
        header=json.dumps(dict(request='abort-me',layer='0',size=len(payload),sha256=hashlib.sha256(payload).hexdigest())).encode()
        with Receiver(timeout=1) as receiver, socket.create_connection(receiver.address) as sock:
            sock.sendall(struct.pack('!I',len(header))+header+payload[:1])
            deadline=time.monotonic()+1
            while not receiver.pending_bytes and time.monotonic()<deadline: time.sleep(.001)
            self.assertEqual(receiver.pending_bytes,len(payload))
            receiver.discard('abort-me'); sock.sendall(payload[1:])
            self.assertEqual(sock.recv(3),b'ERR')
            self.assertEqual(receiver.buffered_bytes,0)
            self.assertNotIn('abort-me',receiver.frames)

    def test_chunked_transfer_and_feedback(self):
        with Receiver(max_bytes=4096) as r:
            with StreamSender(r.address,'r',connections=2) as s:
                s.submit_blob('layer',b'abcd'*100,chunk_bytes=31)
            self.assertEqual(assemble_chunks(r.take('r')),{'layer':b'abcd'*100})
            self.assertEqual(s.stats['bytes'],400)
            self.assertIn('rtt_ms',s.stats)
            monitor=LinkMonitor(1000)
            monitor.record(s.stats)
            sample=monitor.snapshot(queue_depth=2)
            self.assertGreater(sample['bandwidth_bytes_s'],0)
            self.assertEqual(sample['queue_depth'],2)
            monitor.record({'bytes':100,'elapsed_s':1,'failures':1,'retransmissions':1})
            self.assertLess(monitor.snapshot()['bandwidth_bytes_s'],1000)

    def test_missing_chunks_rejected(self):
        with self.assertRaises(ValueError): assemble_chunks({'a:0:2':b'x'})
        with self.assertRaises(ValueError): assemble_chunks({'a:0:1':b'x','a:0:2':b'y'})

    def test_roles_drain_before_switch(self):
        changes=[]; pool=RolePool({'a':'prefill','b':'decode'},lambda node,role:changes.append((node,role)))
        with pool.acquire('a','prefill'):
            pool.reallocate(0)
            self.assertEqual(pool.roles['a'],'prefill')
            with self.assertRaises(RuntimeError):
                with pool.acquire('a','prefill'): pass
        self.assertEqual(pool.roles['a'],'decode')
        self.assertEqual(changes,[('a','decode')])
        with self.assertRaises(ValueError): pool.reallocate(2)
