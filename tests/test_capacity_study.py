import unittest

from prefill_relay.analytics import Profile
from prefill_relay.calibration import select_decode_batch
from prefill_relay.capacity_study import modeled_ttft, study


class CalibrationTests(unittest.TestCase):
    def test_no_promotion_for_fast_incomplete_or_slow_batches(self):
        rows = [dict(batch=batch, length=n, decode_step_seconds=step)
                for batch, n, step in [(1, 10, .024), (1, 80, .025), (2, 10, .01),
                                       (3, 10, .02), (3, 80, .026)]]
        result = select_decode_batch(rows, [10, 80], 40, 3)
        self.assertEqual(result['batch_max'], 1)
        self.assertEqual(result['decode_step_s'], .025)
        self.assertEqual(result['by_batch'][1]['missing_lengths'], [80])
        result = select_decode_batch(rows, [10, 80], 50, 3)
        self.assertEqual(result['batch_max'], 0)
        self.assertIsNone(result['decode_step_s'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            select_decode_batch(rows + rows[:1], [10], 40)

    def test_link_serialization_out_of_order_and_decoder_queue(self):
        profile = Profile([1, 10], [1, 10], [20, 20])
        row = dict(threshold=0, prefill_instances=0, decode_instances=1)
        # Both prefill remotely. Request 1 completes compute first at t=2,
        # transfer at t=4 and occupies decode until t=14. Request 0 arrives
        # at t=0, finishes compute at t=10, transfer at t=12, waits to t=14.
        times = modeled_ttft([10, 1], 1, row, profile, 2, 10, 1, 10)
        self.assertEqual(times, {'mean_s': 8.5, 'p90_s': 14})
        times = modeled_ttft([1, 1], 1, row, profile, 2, 10, 1, 1)
        self.assertEqual(times, {'mean_s': 3.5, 'p90_s': 4})

    def test_budget_baselines_and_no_slo_feasible_output(self):
        profile = dict(profile=dict(lengths=[1, 100], seconds=[.01, .1], kv_bytes=[100, 1000]),
                       rows=[dict(batch=1, length=1, decode_step_seconds=.03)])
        settings = dict(mu=3, sigma=.5, low=1, high=100, max_batch=1, remote_instances=2,
                        local_instances=3, output_tokens=4, samples=32, threshold_step=10,
                        remote_unit_cost=2., local_unit_cost=1., link_gbps=10, offered_fraction=.8, seed=0)
        infeasible = study(profile, [1], 40, settings)
        self.assertEqual(infeasible['policies'], {})
        result = study(profile, [1], 30, settings)
        policies = result['policies']
        self.assertEqual(policies['all_local_same_budget']['resource_cost'], 7)
        self.assertEqual(policies['all_local_same_count']['resource_cost'], 5)
        self.assertEqual(policies['all_local_same_count']['egress_gbps'], 0)
        self.assertEqual(policies['all_remote']['p'], 1)
        self.assertEqual(result['configurations_searched'], 33)
        self.assertEqual(len(result['allocation_sweep']), 3)
        self.assertTrue(all(row['modeled_ttft']['mean_s'] > 0 for row in policies.values()))
