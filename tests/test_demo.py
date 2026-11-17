import importlib.util
import os
import unittest

@unittest.skipUnless(importlib.util.find_spec('torch'),'optional torch dependency is not installed')
class DemoTests(unittest.TestCase):
    def test_end_to_end_paths(self):
        from prefill_relay.demo import run_demo
        result=run_demo(os.environ.get('PREFILL_RELAY_TEST_DEVICE','cpu'),128,32,2,2)
        cases=result['cases']
        self.assertTrue(all(c['correct'] for c in cases.values()))
        self.assertEqual(cases['cold_remote']['layers_streamed'],4)
        self.assertLess(cases['remote_has_longer_prefix']['transferred_bytes'],cases['cold_remote']['transferred_bytes'])
        self.assertTrue(cases['abundant_prefix_migration']['migrated_prefix'])
        self.assertEqual(cases['short_local']['transferred_bytes'],0)
