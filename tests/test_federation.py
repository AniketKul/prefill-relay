from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from prefill_relay.federation import Federation
from prefill_relay.runtime import Coordinator


@unittest.skipUnless(importlib.util.find_spec('torch'), 'requires gpu extra')
class FederationTests(unittest.TestCase):
    device = os.environ.get('PREFILL_RELAY_FEDERATION_DEVICE', 'cpu')

    def test_cross_site_handoff_affinity_outage_and_return_path(self):
        with tempfile.TemporaryDirectory() as root:
            a = Coordinator(Path(root)/'a', device=self.device, bandwidth_bytes_s=200*2**20, threshold=8)
            b = Coordinator(Path(root)/'b', device=self.device, bandwidth_bytes_s=100*2**20, threshold=8)
            with Federation({'a': a, 'b': b}) as service:
                tokens = list(range(48))
                first = service.run(tokens, 3, 'all-remote', pd_site='b')
                self.assertEqual((first['producer_site'], first['pd_site']), ('a', 'b'))
                self.assertEqual(first['worker_pids'][first['producer']], a.workers[first['producer']].info['pid'])
                self.assertEqual(first['worker_pids'][first['decoder']], b.workers[first['decoder']].info['pid'])
                self.assertEqual(first['generated_tokens'], b.reference(tokens, 3)['generated_tokens'])
                self.assertGreater(a.rate.stats()['bytes'], 0)
                self.assertEqual(b.rate.stats()['bytes'], 0)

                # A complete decoder hit pins the global route to b. A cold b
                # decoder can also resume a prefix held by a's producer.
                self.assertIsNone(service.run(tokens, 3)['producer_site'])
                for name in b.roles.available('decode'):
                    b.workers[name].call('clear')
                extended = tokens + [99]
                resumed = service.run(extended, 3, 'all-remote', pd_site='b')
                self.assertEqual(resumed['prefix_tokens'], len(tokens))
                self.assertEqual(resumed['generated_tokens'], a.reference(extended,3)['generated_tokens'])

                for name in b.roles.available('decode'):
                    b.workers[name].call('clear')
                resumed_locally = extended + [23]
                migrated = service.run(resumed_locally, 3, pd_site='b')
                self.assertEqual(migrated['producer_site'], 'b')
                self.assertEqual(migrated['target'], 'local')
                self.assertEqual(migrated['migration']['status'], 'transferred')
                self.assertEqual(migrated['migration']['source'][0], 'remote')
                self.assertEqual(migrated['prefix_tokens'], len(extended))
                self.assertEqual(migrated['generated_tokens'], a.reference(resumed_locally, 3)['generated_tokens'])

                a.set_bandwidth(0)
                a_bytes = a.rate.stats()['bytes']
                other = list(range(80, 110))
                back = service.run(other, 3, 'all-remote', pd_site='a')
                self.assertEqual((back['producer_site'], back['pd_site']), ('b', 'a'))
                self.assertEqual(back['generated_tokens'], a.reference(other,3)['generated_tokens'])
                self.assertGreater(b.rate.stats()['bytes'], 0)
                self.assertEqual(a.rate.stats()['bytes'], a_bytes)
                # Both PD destinations now share the same B producer and egress clock.
                with ThreadPoolExecutor(2) as executor:
                    calls = [executor.submit(service.run, [j] * 38, 2, 'all-remote', pd_site=name)
                             for j, name in [(71, 'a'), (72, 'b')]]
                    both = [future.result(timeout=20) for future in calls]
                self.assertEqual([row['producer_site'] for row in both], ['b', 'b'])
                self.assertEqual([row['pd_site'] for row in both], ['a', 'b'])
                self.assertEqual(both[0]['worker_pids']['remote-0'], both[1]['worker_pids']['remote-0'])
                self.assertTrue(all(row['transfers']==0 for row in b.stats()['workers'].values()))
                b.set_bandwidth(0)
                warm = service.run(resumed_locally, 3)
                self.assertEqual((warm['pd_site'], warm['producer_site'], warm['transferred_bytes']), ('b', None, 0))
                fallback = service.run([6, 2, 3, 4], 2, pd_site='a')
                self.assertEqual((fallback['target'], fallback['producer_site']), ('local', 'a'))


    def test_profile_uses_the_selected_clusters(self):
        def profile(seconds):
            return {'profile': {'lengths': [1, 128], 'seconds': [.001, seconds],
                                'kv_bytes': [2048, 20480]},
                    'rows': [{'batch': 1, 'decode_step_seconds': .02}]}
        with tempfile.TemporaryDirectory() as root:
            a = Coordinator(Path(root)/'a', device=self.device, profile=profile(.2), output_tokens=2)
            b = Coordinator(Path(root)/'b', device=self.device, profile=profile(.5), output_tokens=2)
            with Federation({'a': a, 'b': b}) as service:
                _, controller = service.routes['a', 'b']
                _, producer_profile, pd_profile, _, _ = controller.args
                self.assertEqual(producer_profile.at(128)[0], .5)
                self.assertEqual(pd_profile.at(128)[0], .2)
                a.set_bandwidth(0)
                row = service.run(list(range(64)), 2, pd_site='a')
                self.assertEqual(row['generated_tokens'], b.reference(list(range(64)), 2)['generated_tokens'])
                self.assertIsNotNone(controller.last_long)

    @unittest.skipUnless(os.environ.get('PREFILL_RELAY_FEDERATION_TRAINED') == '1',
                         'enable trained federation validation explicitly')
    def test_trained_cross_site_reference_and_full_hit(self):
        options = dict(backend='hf', device=self.device, model_id='Qwen/Qwen3.5-0.8B',
                       revision='2fc06364715b967f1860aea9cf38778875588b17')
        with tempfile.TemporaryDirectory() as root:
            a = Coordinator(Path(root)/'a', bandwidth_bytes_s=1024*2**20, **options)
            b = Coordinator(Path(root)/'b', bandwidth_bytes_s=512*2**20, **options)
            with Federation({'a': a, 'b': b}) as service:
                tokens = [11, 22, 33, 44] * 16
                expected = b.reference(tokens, 4)['generated_tokens']
                row = service.run(tokens, 4, 'all-remote', pd_site='b')
                self.assertEqual((row['producer_site'], row['pd_site']), ('a', 'b'))
                self.assertEqual(row['generated_tokens'], expected)
                self.assertEqual(row['transferred_bytes'], a.rate.stats()['bytes'])
                self.assertGreater(row['transferred_bytes'], 0)
                warm = service.run(tokens, 4)
                self.assertEqual((warm['pd_site'], warm['producer_site']), ('b', None))
                self.assertEqual(warm['transferred_bytes'], 0)
                self.assertEqual(warm['generated_tokens'], expected)
