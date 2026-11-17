"""Numerical cache-handoff checks; PREFILL_RELAY_TEST_DEVICE=cuda enables H100 checks."""
import hashlib
import json
import os
import struct
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from prefill_relay.tiny_model import TinyHybrid, encode_state, decode_state, delta, merge


@unittest.skipIf(torch is None, 'optional torch dependency is not installed')
class TinyModelTests(unittest.TestCase):
    def setUp(self):
        self.device = os.environ.get('PREFILL_RELAY_TEST_DEVICE', 'cpu')
        self.model = TinyHybrid().to(self.device).eval()
        self.tokens = torch.tensor([3, 8, 6, 9, 12, 14, 7, 18, 21, 4, 11], device=self.device)

    def test_incremental_logits_match_full_recomputation(self):
        logits, full = self.model(self.tokens)
        prefix_logits, prefix = self.model(self.tokens[:6])
        suffix_logits, resumed = self.model(self.tokens[6:], prefix)
        torch.testing.assert_close(torch.cat((prefix_logits, suffix_logits)), logits, rtol=2e-5, atol=2e-6)
        for name in full:
            torch.testing.assert_close(resumed[name], full[name], rtol=2e-5, atol=2e-6)
        # Reusing the checkpoint must not mutate it.
        self.assertEqual(prefix['length'].item(), 6)
        self.assertEqual(prefix['layer.3.key'].shape[0], 6)

    def test_decode_handoff_matches_every_next_token(self):
        _, state = self.model(self.tokens)
        state = decode_state(encode_state(state), self.device)
        tokens = self.tokens
        for next_token in [2, 5, 10, 17]:
            token = torch.tensor([next_token], device=self.device)
            tokens = torch.cat((tokens, token))
            incremental, state = self.model(token, state)
            expected, _ = self.model(tokens)
            torch.testing.assert_close(incremental[-1], expected[-1], rtol=2e-5, atol=2e-6)

    def test_delta_replaces_recurrent_state_and_sends_only_missing_kv(self):
        _, base = self.model(self.tokens[:6])
        _, full = self.model(self.tokens)
        change = delta(full, 6)
        self.assertEqual(change['layer.3.key'].shape[0], 5)
        self.assertEqual(change['layer.0.matrix'].shape, full['layer.0.matrix'].shape)
        restored = merge(base, decode_state(encode_state(change), self.device))
        for name in full:
            torch.testing.assert_close(restored[name], full[name])
        _, wrong_base = self.model(torch.tensor([1, 8, 6, 9, 12, 14], device=self.device))
        with self.assertRaisesRegex(ValueError, 'prefix'):
            merge(wrong_base, change)
        with self.assertRaises(ValueError):
            delta(full, 12)

    def test_empty_base_delta(self):
        _, full = self.model(self.tokens)
        rebuilt = merge(None, delta(full, 0))
        for name in full:
            torch.testing.assert_close(rebuilt[name], full[name])

    def test_layer_callback_is_complete_and_serializable(self):
        observed = []
        _, full = self.model(self.tokens, on_layer=lambda i, state: observed.append((i, decode_state(encode_state(state)))))
        self.assertEqual([i for i, _ in observed], [0, 1, 2, 3])
        assembled = {}
        for i, state in observed:
            assembled.update(state)
            self.assertTrue(any(name.startswith(f'layer.{i}.') for name in state))
        for name in full:
            torch.testing.assert_close(assembled[name], full[name].cpu())

    def test_corrupt_or_truncated_serialization_is_rejected(self):
        _, state = self.model(self.tokens)
        payload = encode_state(state)
        for bad in [payload[:-1], payload + b'x', payload[:4] + b'junk' + payload[8:], payload[:-1] + bytes([payload[-1] ^ 1])]:
            with self.assertRaises(ValueError):
                decode_state(bad)
        with self.assertRaises(ValueError):
            decode_state(payload, max_bytes=32)

    def test_forged_dimensions_are_rejected_before_allocation(self):
        _, state = self.model(self.tokens)
        payload = encode_state(state)
        size = struct.unpack('!I', payload[8:12])[0]
        metadata = json.loads(payload[12:12 + size])
        metadata[0]['shape'] = [2**60, 2**60]
        header = json.dumps(metadata, separators=(',', ':')).encode()
        body = payload[:8] + struct.pack('!I', len(header)) + header + payload[12 + size:-32]
        forged = body + hashlib.sha256(body).digest()
        with self.assertRaisesRegex(ValueError, 'byte count'):
            decode_state(forged)

    def test_model_mismatch_is_rejected(self):
        _, cache = self.model(self.tokens)
        other = TinyHybrid(seed=1).to(self.device)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            other(self.tokens[:1], cache)

    def test_invalid_inputs_and_cache_are_rejected(self):
        with self.assertRaises(ValueError):
            self.model(torch.tensor([], dtype=torch.long, device=self.device))
        with self.assertRaises(ValueError):
            self.model(torch.tensor([256], device=self.device))
        _, state = self.model(self.tokens)
        del state['layer.1.matrix']
        with self.assertRaises(ValueError):
            self.model(self.tokens[:1], state)


if __name__ == '__main__':
    unittest.main()
