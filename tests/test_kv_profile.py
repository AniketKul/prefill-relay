import importlib.util
from types import SimpleNamespace
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'install the gpu extra for tensor checks')
class CacheSizeTests(unittest.TestCase):
    def test_counts_full_and_bounded_state_with_mixed_dtypes(self):
        import torch
        from prefill_relay.kv_profile import cache_sizes
        full = SimpleNamespace(keys=torch.zeros(1, 2, 7, 4, dtype=torch.bfloat16),
                               values=torch.zeros(1, 2, 7, 4, dtype=torch.bfloat16))
        linear = SimpleNamespace(conv_states=[torch.zeros(1, 3, 4, dtype=torch.bfloat16)],
                                 recurrent_states=[torch.zeros(1, 2, 4, 4, dtype=torch.float32)])
        cache = SimpleNamespace(layers=[full, linear])
        kinds = ['full_attention', 'linear_attention']
        self.assertEqual(cache_sizes(cache, kinds, 7), (224, 152))
        with self.assertRaisesRegex(ValueError, 'prompt length'):
            cache_sizes(cache, kinds, 6)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            cache_sizes(cache, kinds[:1], 7)
