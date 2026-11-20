import importlib.util
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires gpu extra')
class InFlightLinkTests(unittest.TestCase):
    def test_placement_sees_partial_transfer_before_prefill_returns(self):
        from prefill_relay.runtime import Coordinator

        device = os.environ.get('PREFILL_RELAY_TEST_DEVICE', 'cpu')
        with tempfile.TemporaryDirectory() as path, Coordinator(path, device=device) as service:
            tokens = list(range(67))
            # Hold the large full-attention frame after the smaller recurrent layers start.
            full_frame, release = threading.Event(), threading.Event()
            reserve = service.rate._reserve
            def gated_reserve(size):
                if size > 16000:
                    full_frame.set()
                    if not release.wait(4):
                        raise RuntimeError('test gate timed out')
                return reserve(size)
            service.rate._reserve = gated_reserve
            with ThreadPoolExecutor(1) as executor:
                request = executor.submit(service.run, tokens, 2, 'all-remote')
                try:
                    self.assertTrue(full_frame.wait(3))
                    deadline = time.monotonic() + 1
                    while True:
                        snapshot = service.workers['remote-0'].call('link')
                        if snapshot.get('frames_acked', 0) or time.monotonic() >= deadline:
                            break
                        time.sleep(.01)
                    self.assertGreater(snapshot['frames_acked'], 0)
                    with service.control_lock:
                        _, _, sample = service._choose(tokens, 'all-remote')
                    self.assertGreater(sample['utilization'], 0)
                    self.assertFalse(request.done())
                finally:
                    release.set()
                self.assertEqual(request.result()['target'], 'remote')
            self.assertEqual(service.workers['remote-0'].call('link'), {})
