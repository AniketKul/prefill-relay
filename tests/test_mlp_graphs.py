"""Opt-in graph replay preserves eager outputs and retained tensors across calls."""
import os
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and os.environ.get('PREFILL_RELAY_TEST_DEVICE') == 'cuda',
                     'requires opt-in CUDA test device')
class MLPGraphTests(unittest.TestCase):
    def test_cache_reuse_and_retained_outputs_across_streams(self):
        from prefill_relay.mlp_graphs import DecodeMLPGraphs
        device = torch.device('cuda', torch.cuda.current_device())
        mlp = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 16)).to(device).eval()
        eager = mlp.forward
        layer = SimpleNamespace(mlp=mlp)
        backend = SimpleNamespace(device=device, model=SimpleNamespace(training=False, model=SimpleNamespace(layers=[layer])))
        graphs = DecodeMLPGraphs(backend)
        with torch.inference_mode():
            x = torch.randn(2, 1, 16, device=device)
            y = x * 3
            x_ref, y_ref = eager(x), eager(y)
            mlp(x)
            self.assertFalse(graphs.graphs)
            with graphs.enabled():
                first = mlp(x)
            other = torch.cuda.Stream(device=device)
            with torch.cuda.stream(other), graphs.enabled():
                second = mlp(y)
            torch.cuda.current_stream(device).wait_stream(other)
            torch.testing.assert_close(first, x_ref, rtol=0, atol=0)
            torch.testing.assert_close(second, y_ref, rtol=0, atol=0)
            self.assertEqual(len(graphs.graphs), 1)
            long_input = x.expand(2, 3, 16).contiguous()
            with graphs.enabled():
                torch.testing.assert_close(mlp(long_input), eager(long_input), rtol=0, atol=0)
            self.assertEqual(len(graphs.graphs), 1)


@unittest.skipUnless(torch is not None and os.environ.get('PREFILL_RELAY_TRAINED_MODEL') and
                     os.environ.get('PREFILL_RELAY_TEST_DEVICE') == 'cuda', 'requires trained CUDA model')
class TrainedMLPGraphTests(unittest.TestCase):
    def test_batches_preserve_all_decode_logits_and_checkpoint(self):
        from prefill_relay.hf_backend import HFHybrid
        from prefill_relay.hf_decode import generate_continuation
        backend = HFHybrid(os.environ['PREFILL_RELAY_TRAINED_MODEL'], device='cuda')
        backend.enable_decode_graphs()
        for batch in (1, 2, 3, 4):
            inputs = torch.arange(12 * batch, device=backend.device).reshape(batch, 12) + 32
            _, state = backend.prefill(inputs[0] if batch == 1 else inputs)
            before = {k: v.clone() for k, v in state.items()}
            def collect(generate):
                logits = []
                hook = backend.model.register_forward_hook(lambda m, a, r: logits.append(r.logits.clone()))
                try:
                    return generate(), logits
                finally:
                    hook.remove()
            expected, eager = collect(lambda: generate_continuation(backend, state, 8))
            actual, graphed = collect(lambda: backend.generate(state, 8))
            self.assertTrue(torch.equal(expected, actual))
            self.assertEqual(len(eager), 7)
            self.assertEqual(len(graphed), 7)
            for a, b in zip(eager, graphed):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key, source in before.items():
                self.assertTrue(torch.equal(source, state[key]))
        self.assertEqual(len(backend._decode_graphs.graphs), 24 * 4)
