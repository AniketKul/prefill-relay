import importlib.util
import os
import tempfile
import unittest


@unittest.skipUnless(importlib.util.find_spec('torch'),'requires gpu extra')
class RuntimeTests(unittest.TestCase):
    def test_profile_feedback_applies_outage_role_change(self):
        from prefill_relay.runtime import Coordinator
        profile={'profile':{'lengths':[1,128],'seconds':[.001,.1],'kv_bytes':[100,1000]},
                 'rows':[{'batch':1,'decode_step_seconds':.1}]}
        with tempfile.TemporaryDirectory() as path, Coordinator(path,device=os.environ.get('PREFILL_RELAY_TEST_DEVICE','cpu'),profile=profile,output_tokens=2) as service:
            remote=service.run(list(range(32)),2)
            self.assertEqual(remote['target'],'remote')
            service.set_bandwidth(0)
            tokens=list(range(32,64))
            local=service.run(tokens,2)
            self.assertEqual(local['target'],'local')
            self.assertIn('prefill',service.roles.roles.values())
            self.assertEqual(local['generated_tokens'],service.reference(tokens,2)['generated_tokens'])

    def test_evicted_migration_base_falls_back_to_real_prefill(self):
        from unittest.mock import patch
        from prefill_relay.runtime import Coordinator
        tokens = list(range(48))
        device = os.environ.get('PREFILL_RELAY_TEST_DEVICE', 'cpu')
        with tempfile.TemporaryDirectory() as path, Coordinator(path, device=device, threshold=64,
                                                                bandwidth_bytes_s=1000*2**20) as service:
            for length in (16, 32):
                service.run(tokens[:length], 2, 'all-remote')
            destination = service.directory._client(('local','local-0'), service.namespace)
            self.assertTrue(destination.delete(tokens[:32]))
            self.assertEqual(destination.match_length(tokens), 16)
            apply_delta = destination.apply_delta
            def evict_then_apply(prompt, delta, known):
                self.assertTrue(destination.delete(tokens[:16]))
                return apply_delta(prompt, delta, known)
            with patch.object(destination, 'apply_delta', side_effect=evict_then_apply):
                result = service.run(tokens, 2)
            self.assertEqual(result['migration']['status'], 'stale_prefix')
            self.assertEqual(result['migration']['length'], 0)
            self.assertGreater(result['transferred_bytes'], 0)
            self.assertEqual(result['target'], 'local')
            self.assertEqual(result['prefix_tokens'], 0)
            self.assertEqual(result['generated_tokens'], service.reference(tokens,2)['generated_tokens'])
            self.assertTrue(all(row['transfers'] == 0 for row in service.stats()['workers'].values()))

    def test_small_cache_admission_does_not_abort_generation(self):
        from prefill_relay.runtime import Coordinator
        with tempfile.TemporaryDirectory() as path, Coordinator(path,device=os.environ.get('PREFILL_RELAY_TEST_DEVICE','cpu'),cache_bytes=4096) as service:
            tokens=list(range(32))
            # Receiver capacity is independent from optional reusable-cache admission.
            for worker in service.workers.values():
                self.assertEqual(worker.call('stats')['capacity_blocks'],1)
            expected=service.reference(tokens,2)['generated_tokens']
            result=service.run(tokens,2,'all-remote')
            self.assertEqual(result['generated_tokens'],expected)
            self.assertFalse(result['prefill']['stored'])
            self.assertTrue(all(row['transfers']==0 for row in service.stats()['workers'].values()))
            from prefill_relay.transport import Receiver
            address=service.workers['local-0'].info['receiver']
            with Receiver(max_bytes=1) as refused:
                service.workers['local-0'].info['receiver']=refused.address
                try:
                    with self.assertRaises(RuntimeError): service.run(tokens,2,'all-remote')
                finally: service.workers['local-0'].info['receiver']=address
            self.assertGreater(service.monitor.snapshot()['failures'],0)
            self.assertTrue(all(row['transfers']==0 for row in service.stats()['workers'].values()))

    def test_process_handoff_full_hit_migration_and_roles(self):
        from prefill_relay.runtime import Coordinator
        tokens=[(i*17+3)%256 for i in range(67)]
        with tempfile.TemporaryDirectory() as path, Coordinator(path,device=os.environ.get('PREFILL_RELAY_TEST_DEVICE','cpu'),threshold=32) as service:
            expected=service.reference(tokens,4)['generated_tokens']
            remote=service.run(tokens,4,'all-remote')
            self.assertEqual(remote['generated_tokens'],expected)
            self.assertEqual(len(set(remote['worker_pids'].values())),2)
            self.assertEqual(len(remote['prefill']['layer_ready_s']),4)
            self.assertIsNotNone(remote['prefill']['sender']['first_send_at'])
            self.assertGreater(remote['transferred_bytes'],0)
            warm=service.run(tokens,4,'all-remote')
            self.assertEqual(warm['prefix_tokens'],67)
            self.assertEqual(warm['destination_prefix_tokens'],67)
            self.assertEqual(warm['generated_tokens'],expected)
            local=service.run(tokens,4,'all-local')
            self.assertEqual(local['generated_tokens'],expected)
            self.assertEqual(local['target'],'local')
            self.assertEqual(local['transferred_bytes'],0)
            service.set_bandwidth(0)
            fallback=service.run(tokens,4)
            self.assertEqual(fallback['target'],'local')
            stats=service.stats()
            self.assertTrue(stats['role_history'])
            self.assertTrue(all(s['transfers']==0 for s in stats['workers'].values()))
            service.reset_caches()
            self.assertTrue(all(s['checkpoints']==0 for s in service.stats()['workers'].values()))


if __name__=='__main__': unittest.main()
