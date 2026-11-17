"""Native cache ownership tests, plus optional trained generation comparisons."""
import os
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from prefill_relay.hf_decode import generate_continuation


if torch is not None:
    class FakeModel(torch.nn.Module):
        def __init__(self, vocab):
            super().__init__()
            self.vocab, self.calls = vocab, 0
            self.invalid_output = None

        def forward(self, input_ids, past_key_values, use_cache, logits_to_keep):
            assert torch.is_inference_mode_enabled()
            assert use_cache and logits_to_keep == 1
            assert torch.equal(input_ids[:, 0], past_key_values.value)
            self.calls += 1
            following = (past_key_values.value + 1) % self.vocab
            # Returning a distinct object catches callers that ignore returned caches.
            new_cache = SimpleNamespace(value=following.clone())
            past_key_values.value.fill_(-99)
            logits = torch.nn.functional.one_hot(following, self.vocab).float()[:, None]
            if self.invalid_output == 'shape':
                logits = logits[:, 0]
            elif self.invalid_output == 'nonfinite':
                logits.fill_(float('nan'))
            return SimpleNamespace(logits=logits, past_key_values=None if self.invalid_output == 'cache' else new_cache)


    class FakeBackend:
        device = torch.device('cpu')
        vocab_size = 8
        max_context = 32

        def __init__(self):
            self.restore_calls = 0
            self.model = FakeModel(self.vocab_size)

        def _restore(self, state, batch):
            self.restore_calls += 1
            if not torch.equal(state['model_config'], torch.tensor([7])):
                raise ValueError('different model cache')
            return SimpleNamespace(value=state['latent'].clone())


def fake_state(batch):
    first = torch.arange(1, batch + 1) % 8
    logits = torch.nn.functional.one_hot(first, 8).float()[:, None]
    return {'length': torch.tensor(4), 'tokens': torch.arange(4) if batch == 1 else torch.arange(4).repeat(batch, 1),
            'model_config': torch.tensor([7]), 'latent': first,
            'last_logits': logits[0] if batch == 1 else logits}


@unittest.skipIf(torch is None, 'optional torch dependency is not installed')
class NativeDecodeTests(unittest.TestCase):
    def test_single_restore_reused_cache_and_immutable_checkpoint(self):
        for batch in [1, 2, 4]:
            backend, state = FakeBackend(), fake_state(batch)
            before = {k: v.clone() for k, v in state.items()}
            result = generate_continuation(backend, state, 6)
            expected = (torch.arange(1, batch + 1)[:, None] + torch.arange(6)) % 8
            torch.testing.assert_close(result, expected)
            self.assertEqual(result.shape, (batch, 6))
            self.assertEqual(backend.restore_calls, 1)
            self.assertEqual(backend.model.calls, 5)
            for name in state:
                torch.testing.assert_close(state[name], before[name], rtol=0, atol=0)

    def test_one_token_uses_cached_logits_without_model_call(self):
        backend = FakeBackend()
        self.assertEqual(generate_continuation(backend, fake_state(1), 1).tolist(), [[1]])
        self.assertEqual(backend.restore_calls, 1)
        self.assertEqual(backend.model.calls, 0)

    def test_invalid_count_context_identity_and_logits(self):
        for count in [0, -1, True, 1.5, 30]:
            with self.assertRaises(ValueError):
                generate_continuation(FakeBackend(), fake_state(1), count)
        state = fake_state(1)
        state['model_config'][0] = 9
        with self.assertRaisesRegex(ValueError, 'different model'):
            generate_continuation(FakeBackend(), state, 2)
        for value in [torch.zeros(8), torch.ones(1, 8, dtype=torch.int64), torch.full((1, 8), float('nan'))]:
            state = fake_state(1)
            state['last_logits'] = value
            with self.assertRaises(ValueError):
                generate_continuation(FakeBackend(), state, 2)
        state = fake_state(1)
        state['tokens'] = torch.zeros(4)
        with self.assertRaises(ValueError):
            generate_continuation(FakeBackend(), state, 2)

    def test_model_must_return_valid_logits_and_native_cache(self):
        for kind in ['cache', 'shape', 'nonfinite']:
            backend = FakeBackend()
            backend.model.invalid_output = kind
            with self.assertRaises(ValueError):
                generate_continuation(backend, fake_state(1), 2)


@unittest.skipUnless(torch is not None and os.environ.get('PREFILL_RELAY_TRAINED_MODEL'),
                     'set PREFILL_RELAY_TRAINED_MODEL to run trained decode checks')
class TrainedNativeDecodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from prefill_relay.hf_backend import HFHybrid
        cls.backend = HFHybrid(os.environ['PREFILL_RELAY_TRAINED_MODEL'],
                               device=os.environ.get('PREFILL_RELAY_TEST_DEVICE', 'cuda'))

    def test_trained_tokens_every_step_logits_and_checkpoint_match(self):
        backend = self.backend
        prompts = [backend.tokenizer(
            f'The capital of France is Paris. The capital of {country} is',
            return_tensors='pt').input_ids[0].to(backend.device)
            for country in ('Japan', 'Germany', 'Italy', 'Spain')]
        self.assertEqual(len({p.numel() for p in prompts}), 1)
        for batch in [1, 2, 4]:
            tokens = prompts[0] if batch == 1 else torch.stack(prompts[:batch])
            _, state = backend.prefill(tokens)
            snapshot = {k: v.clone() for k, v in state.items()}
            original_restore = backend._restore
            restores = []
            def counted_restore(*args, **kwargs):
                restores.append(1)
                return original_restore(*args, **kwargs)
            observed = []
            hook = backend.model.register_forward_hook(lambda module, args, output: observed.append(output.logits.clone()))
            backend._restore = counted_restore
            try:
                actual = generate_continuation(backend, state, 8)
            finally:
                backend._restore = original_restore
                hook.remove()
            self.assertEqual(len(restores), 1)
            self.assertEqual(len(observed), 7)
            self.assertEqual(actual.shape, (batch, 8))
            logits, cache, expected, reference_logits = state['last_logits'], state, [], []
            for index in range(8):
                token = logits[-1].argmax().reshape(1) if batch == 1 else logits[:, -1].argmax(-1)
                expected.append(token)
                if index != 7:
                    logits, cache = backend.forward(token if batch == 1 else token[:, None], cache)
                    reference_logits.append(logits[None] if batch == 1 else logits)
            torch.testing.assert_close(actual, torch.stack(expected, dim=1), rtol=0, atol=0)
            for native, reference in zip(observed, reference_logits):
                torch.testing.assert_close(native, reference, rtol=0, atol=0)
            for key in state:
                torch.testing.assert_close(state[key], snapshot[key], rtol=0, atol=0)
