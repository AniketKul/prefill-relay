"""Opt-in trained-model checks; no weights downloaded by ordinary test runs."""
import os
import unittest

_MODEL = os.environ.get('PREFILL_RELAY_TRAINED_MODEL')


@unittest.skipUnless(_MODEL, 'set PREFILL_RELAY_TRAINED_MODEL to run pretrained model checks')
class TrainedBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from prefill_relay.hf_backend import HFHybrid
        cls.backend = HFHybrid(_MODEL, device=os.environ.get('PREFILL_RELAY_TEST_DEVICE', 'cuda'))

    def test_trained_layer_stream_delta_prefix_full_hit_and_batched_handoff(self):
        from prefill_relay.hf_validate import validate_backend
        result = validate_backend(self.backend)
        self.assertEqual(result['handoff_max_logit_error'], 0)
        self.assertTrue(result['wire_state_equal'])
        self.assertTrue(result['delta_state_equal'])
        self.assertEqual(result['full_hit_model_calls'], 0)

    def test_invalid_inputs_and_foreign_state(self):
        import torch
        b = self.backend
        with self.assertRaises(ValueError):
            b.forward(torch.tensor([-1], device=b.device))
        with self.assertRaises(ValueError):
            b.forward(torch.tensor([b.vocab_size], device=b.device))
        with self.assertRaises(ValueError):
            b.forward(torch.empty(0, dtype=torch.int64, device=b.device))
        tokens = torch.tensor([1, 2, 3, 4], device=b.device)
        _, state = b.forward(tokens)
        state['model_config'] = state['model_config'].clone()
        state['model_config'][0] ^= 1
        with self.assertRaisesRegex(ValueError, 'different model'):
            b.forward(tokens, state)
