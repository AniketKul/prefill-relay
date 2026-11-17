import json
from pathlib import Path
import unittest
from prfaas.__main__ import search

class ExperimentTests(unittest.TestCase):
    def test_example_search_includes_baselines(self):
        config=json.loads((Path(__file__).resolve().parents[1]/'config/example.json').read_text())
        result,rows=search(config)
        self.assertGreater(result['prfaas']['throughput'],0)
        self.assertEqual(result['homogeneous']['p'],0)
        self.assertEqual(result['naive_heterogeneous']['p'],1)
        self.assertEqual(result['naive_heterogeneous']['prefill_instances'],0)
        self.assertIn('ILLUSTRATIVE',result['provenance'])
        del config['provenance']
        with self.assertRaises(ValueError): search(config)
