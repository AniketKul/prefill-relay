"""Wire format checks independent of downloaded weights."""
import unittest
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from prefill_relay import hf_state as wire


@unittest.skipIf(torch is None, 'optional torch dependency is not installed')
class TrainedStateTests(unittest.TestCase):
    def state(self, length=5, batch=None):
        shape = (length,) if batch is None else (batch, length)
        return {'length': torch.tensor(length), 'tokens': torch.arange(length).expand(shape).clone(),
                'model_config': torch.arange(32, dtype=torch.uint8),
                'layer.3.key': torch.ones((batch or 1, 2, length, 8), dtype=torch.bfloat16),
                'layer.3.value': torch.ones((batch or 1, 2, length, 8), dtype=torch.float16),
                'layer.0.recurrent': torch.ones((batch or 1, 4, 8, 8)),
                'layer.0.conv': torch.ones((batch or 1, 12, 4), dtype=torch.bfloat16),
                'last_logits': torch.ones((batch or 1, 1, 32), dtype=torch.bfloat16)}

    def test_codec_preserves_bf16_fp16_float32_and_zero_dimensions(self):
        state = self.state()
        state['empty'] = torch.empty(1, 2, 0, 8, dtype=torch.bfloat16)
        decoded = wire.decode(wire.encode(state))
        for name in state:
            torch.testing.assert_close(decoded[name], state[name], rtol=0, atol=0)

    def test_checksum_and_byte_limit(self):
        payload = wire.encode(self.state())
        with self.assertRaisesRegex(ValueError, 'checksum'):
            wire.decode(payload[:-1] + bytes([payload[-1] ^ 1]))
        with self.assertRaises(ValueError):
            wire.decode(payload, max_bytes=len(payload) - 1)
        with self.assertRaises(ValueError):
            wire.encode(self.state(), max_bytes=10)
        for payload in [b'', b'PRFHF001', payload[:43], payload[:-10]]:
            with self.assertRaises(ValueError):
                wire.decode(payload)

    def test_batched_delta_slices_all_attention_but_replaces_linear(self):
        for batch in [None, 3]:
            base, full = self.state(2, batch), self.state(5, batch)
            full['layer.0.recurrent'] *= 7
            full['layer.7.key'] = full['layer.3.key'].clone()
            full['layer.7.value'] = full['layer.3.value'].clone()
            base['layer.7.key'] = base['layer.3.key'].clone()
            base['layer.7.value'] = base['layer.3.value'].clone()
            change = wire.delta(full, 2)
            self.assertEqual(change['layer.7.key'].shape[-2], 3)
            self.assertTrue(torch.equal(change['layer.0.recurrent'], full['layer.0.recurrent']))
            restored = wire.merge(base, wire.decode(wire.encode(change)))
            for name in full:
                torch.testing.assert_close(restored[name], full[name])

    def test_wrong_prefix_and_wrong_model_are_rejected(self):
        base, full = self.state(2), self.state(5)
        change = wire.delta(full, 2)
        base['tokens'][0] = 99
        with self.assertRaisesRegex(ValueError, 'prefix identity'):
            wire.merge(base, change)
        base = self.state(2)
        base['model_config'][0] = 99
        with self.assertRaisesRegex(ValueError, 'different model'):
            wire.merge(base, change)
        with self.assertRaises(ValueError):
            wire.merge(None, change)

    def test_empty_destination_and_full_hit(self):
        state = self.state()
        for base, prefix in [(None, 0), (state, 5)]:
            restored = wire.merge(base, wire.delta(state, prefix))
            for name in state:
                torch.testing.assert_close(restored[name], state[name])
