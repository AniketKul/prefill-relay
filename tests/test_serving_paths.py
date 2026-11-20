import importlib.util
import os
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires torch')
class ServingPaths(unittest.TestCase):
    def test_same_tokens_different_kv_repairs_only_changed_prefix_tensor(self):
        import torch
        from prefill_relay.backend import TinyBackend, handoff_delta, merge_handoff, prefix_fingerprints
        backend = TinyBackend()
        _, prefix = backend.prefill(torch.arange(8))
        _, state = backend.prefill(torch.arange(8, 13), prefix)
        bad = dict(prefix)
        bad['layer.3.key'] = prefix['layer.3.key'] + 1
        change = handoff_delta(backend, state, 8, prefix_fingerprints(backend, bad, 8))
        self.assertEqual([k for k in change if k.startswith('replacement.')], ['replacement.layer.3.key'])
        restored = merge_handoff(backend, bad, change)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, restored[key]), key)

    def test_saturated_decoder_replicates_hot_checkpoint_to_idle_decoder(self):
        from prefill_relay.runtime import Coordinator
        with tempfile.TemporaryDirectory() as path, Coordinator(path, max_batch=1) as service:
            tokens = list(range(20))
            first = service.run(tokens, 2, 'all-remote')
            with service.roles.acquire(first['decoder'], 'decode'):
                other = service.run(tokens, 2, 'all-remote')
            self.assertNotEqual(first['decoder'], other['decoder'])
            self.assertEqual(other['handoff_transport'], 'cache')
            self.assertEqual(other['migration']['status'], 'transferred')
            self.assertIsNone(other['producer'])
            self.assertEqual(other['generated_tokens'], first['generated_tokens'])
            self.assertFalse(service.replications)

    def test_graph_mode_must_match_local_device_and_decode_profile(self):
        from prefill_relay.runtime import Coordinator
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError, 'trained CUDA'):
                Coordinator(path, backend='hf', device='cpu', decode_graphs=True)
            with self.assertRaisesRegex(ValueError, 'same decode graph mode'):
                Coordinator(path, backend='hf', device='cuda', profile={'decode_graphs': True})
            with self.assertRaisesRegex(ValueError, 'same decode graph mode'):
                Coordinator(path, backend='hf', device='cuda', profile={}, decode_graphs=True)
            with self.assertRaisesRegex(ValueError, 'requires decode graphs'):
                Coordinator(path, reserve_decode_kv=True)
            with self.assertRaisesRegex(ValueError, 'KV allocation'):
                Coordinator(path, backend='hf', device='cuda', decode_graphs=True,
                            reserve_decode_kv=True, profile={'decode_graphs': True})
            ready = Coordinator(path, backend='hf', device='cuda', decode_graphs=True,
                                profile={'measurements': {'decode_graphs': True}})
            self.assertTrue(ready.config['decode_graphs'])

    def test_decode_cap_covers_every_batch_size_the_runtime_can_emit(self):
        from prefill_relay.runtime import Coordinator
        profile = {'profile':{'lengths':[128]}, 'rows':[
            {'batch':b,'length':128,'decode_step_seconds':t} for b,t in [(1,.02),(2,.03),(3,.01)]]}
        with tempfile.TemporaryDirectory() as path:
            service = Coordinator(path,profile=profile,decode_slo_tps=40,max_batch=3)
            self.assertEqual(service.config['max_batch'],1)
            profile['rows'][0]['decode_step_seconds']=.04
            with self.assertRaisesRegex(ValueError,'contiguous'):
                Coordinator(path,profile=profile,decode_slo_tps=40,max_batch=3)

    @unittest.skipUnless(os.environ.get('PREFILL_RELAY_RDMA_ADDRESS'), 'requires a local RDMA interface')
    def test_local_rdma_and_remote_tcp_handoff(self):
        from prefill_relay.runtime import Coordinator
        with tempfile.TemporaryDirectory() as path, Coordinator(path,
                device=os.environ.get('PREFILL_RELAY_TEST_DEVICE','cpu'),
                local_rdma_address=os.environ['PREFILL_RELAY_RDMA_ADDRESS']) as service:
            tokens = list(range(31))
            expected = service.reference(tokens, 3)['generated_tokens']
            for policy, transport in [('all-local','rdma'), ('all-remote','tcp')]:
                service.reset_caches()
                row = service.run(tokens, 3, policy)
                self.assertEqual(row['handoff_transport'], transport)
                self.assertEqual(row['generated_tokens'], expected)
                self.assertGreater(row['prefill']['sender']['bytes'], 0)
