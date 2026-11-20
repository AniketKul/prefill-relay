import os
import tempfile
import unittest

from prefill_relay.runtime import Coordinator


@unittest.skipUnless(os.environ.get('PREFILL_RELAY_SSH_HOST') and
                     os.environ.get('PREFILL_RELAY_TRAINED_MODEL') == 'Qwen/Qwen3.5-0.8B',
                     'requires an SSH producer and the trained Qwen3.5-0.8B checkpoint')
class TrainedHandoffTests(unittest.TestCase):
    def test_cross_host_cold_and_warm_match_unsplit_producer(self):
        tokens = [760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 6124, 369]
        with tempfile.TemporaryDirectory() as path, Coordinator(
            path, backend='hf', model_id='Qwen/Qwen3.5-0.8B',
            revision='2fc06364715b967f1860aea9cf38778875588b17', device='cpu',
            remote_host=os.environ['PREFILL_RELAY_SSH_HOST'],
            remote_device=os.environ.get('PREFILL_RELAY_REMOTE_DEVICE', 'cuda'),
            remote_project=os.environ.get('PREFILL_RELAY_SSH_PROJECT', 'prefill-relay'),
        ) as service:
            reference = service.reference(tokens, 8)['generated_tokens']
            cold = service.run(tokens, 8, 'all-remote')
            warm = service.run(tokens, 8, 'all-remote')
            self.assertEqual(cold['generated_tokens'], reference)
            self.assertEqual(warm['generated_tokens'], reference)
            self.assertGreater(cold['transferred_bytes'], 0)
            self.assertEqual(warm['transferred_bytes'], 0)
