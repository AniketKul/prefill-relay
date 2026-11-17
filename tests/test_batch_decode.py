from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest

try:
    import torch
    from prefill_relay.batch_decode import BatchDecoder
except ImportError:
    torch = None


class FakeBackend:
    sequence_axis = 2
    device = 'cpu'
    vocab_size = 8
    max_context = 64

    def __init__(self):
        self.identity = torch.tensor([1, 2, 3], dtype=torch.uint8)
        self.calls = []
        self.fail = False

    def state(self, seed=0, length=4):
        logits = torch.zeros(1, self.vocab_size)
        logits[0, seed] = 1
        return dict(length=torch.tensor(length), tokens=torch.arange(length),
                    model_config=self.identity.clone(), last_logits=logits,
                    **{'layer.0.key': torch.zeros(1, 1, length, 2),
                       'layer.1.recurrent': torch.zeros(1, 1, 2, 2)})

    def forward(self, tokens, cache):
        batch = 1 if tokens.ndim == 1 else tokens.shape[0]
        self.calls.append(batch)
        if self.fail: raise RuntimeError('backend failed')
        if cache['tokens'].ndim != tokens.ndim: raise AssertionError('cache/input batch rank mismatch')
        logits = torch.zeros(batch, 1, self.vocab_size)
        for index, token in enumerate(tokens.reshape(-1)):
            logits[index, 0, (int(token)+1) % self.vocab_size] = 1
        state = dict(cache)
        state['length'] = cache['length'] + 1
        state['tokens'] = torch.cat((cache['tokens'], tokens), dim=-1)
        state['last_logits'] = logits[0] if tokens.ndim == 1 else logits
        return state['last_logits'], state


@unittest.skipIf(torch is None, 'optional torch dependency is not installed')
class BatchDecodeTests(unittest.TestCase):
    def simultaneous(self, batcher, states, outputs=4):
        barrier = threading.Barrier(len(states))
        def run(state):
            barrier.wait(timeout=5)
            return batcher.decode(state, outputs)
        with ThreadPoolExecutor(max_workers=len(states)) as executor:
            return list(executor.map(run, states))

    def test_compatible_requests_use_real_batch_shape(self):
        backend = FakeBackend()
        batcher = BatchDecoder(backend, threading.Lock(), max_batch=4, wait_s=.05)
        try:
            rows = self.simultaneous(batcher, [backend.state(i) for i in range(4)])
            self.assertEqual(backend.calls, [4, 4, 4])
            for seed, row in enumerate(rows):
                self.assertEqual(row['generated_tokens'], [(seed+i) % 8 for i in range(4)])
                self.assertEqual(row['batch_size'], 4)
                self.assertGreater(row['decode_s'], 0)
                self.assertGreaterEqual(row['batch_queue_s'], 0)
        finally: batcher.close()

    def test_lengths_and_identity_do_not_mix(self):
        backend = FakeBackend()
        batcher = BatchDecoder(backend, threading.Lock(), wait_s=.03)
        try:
            rows = self.simultaneous(batcher, [backend.state(length=n) for n in (4, 5, 4, 5)], outputs=2)
            self.assertEqual(backend.calls, [2, 2])
            self.assertTrue(all(row['batch_size'] == 2 for row in rows))
            corrupt = backend.state()
            corrupt['model_config'][0] = 9
            with self.assertRaisesRegex(ValueError, 'different model'): batcher.decode(corrupt, 2)
            with self.assertRaisesRegex(ValueError, 'context limit'): batcher.decode(backend.state(), 62)
        finally: batcher.close()

    def test_batch_failure_propagates_and_worker_recovers(self):
        backend = FakeBackend(); backend.fail = True
        batcher = BatchDecoder(backend, threading.Lock(), max_batch=2, wait_s=.02)
        barrier = threading.Barrier(2)
        def run():
            barrier.wait(timeout=5)
            with self.assertRaisesRegex(RuntimeError, 'backend failed'):
                batcher.decode(backend.state(), 2)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(lambda _: run(), range(2)))
            backend.fail = False
            self.assertEqual(batcher.decode(backend.state(), 2)['generated_tokens'], [0, 1])
        finally: batcher.close()
        with self.assertRaisesRegex(RuntimeError, 'closed'): batcher.decode(backend.state(), 2)

    def test_tiny_backend_falls_back_to_single_requests(self):
        backend = FakeBackend(); backend.sequence_axis = 0
        batcher = BatchDecoder(backend, threading.Lock(), wait_s=.02)
        try:
            rows = self.simultaneous(batcher, [backend.state(i) for i in range(3)], outputs=2)
            self.assertEqual(backend.calls, [1, 1, 1])
            self.assertTrue(all(row['batch_size'] == 1 for row in rows))
        finally: batcher.close()

    def test_bounded_queue_and_close_drain(self):
        backend = FakeBackend()
        entered = threading.Event(); lock = threading.Lock(); lock.acquire()
        class ComputeLock:
            def __enter__(self): entered.set(); lock.acquire()
            def __exit__(self, *args): lock.release()
        batcher = BatchDecoder(backend, ComputeLock(), max_batch=1, max_pending=1)
        executor = ThreadPoolExecutor(max_workers=3)
        holding = True
        try:
            first = executor.submit(batcher.decode, backend.state(), 2)
            self.assertTrue(entered.wait(5))
            second = executor.submit(batcher.decode, backend.state(), 2)
            deadline = time.monotonic()+5
            while not batcher.queue and time.monotonic() < deadline: time.sleep(.001)
            self.assertEqual(len(batcher.queue), 1)
            with self.assertRaisesRegex(RuntimeError, 'queue is full'):
                batcher.decode(backend.state(), 2)
            closing = executor.submit(batcher.close)
            lock.release()
            holding = False
            self.assertEqual(first.result(timeout=5)['generated_tokens'], [0, 1])
            self.assertEqual(second.result(timeout=5)['generated_tokens'], [0, 1])
            closing.result(timeout=5)
        finally:
            if holding: lock.release()
            batcher.close()
            executor.shutdown(wait=True)
        self.assertFalse(batcher.thread.is_alive())


if __name__ == '__main__': unittest.main()
