import unittest

from prefill_relay.analytics import Profile, Workload
from prefill_relay.scheduler import Controller
from prefill_relay.stage_load import StageLoad


class StageLoadTests(unittest.TestCase):
    def test_occupancy_batch_queue_and_role_changes(self):
        now = [0.]
        meter = StageLoad({'p': 'local_prefill', 'd': 'decode'}, clock=lambda: now[0])
        with meter.working('d'):
            meter.waiting('r', ('decode',))
            now[0] = 1.
            with meter.working('d'):
                now[0] = 2.
            meter.waiting('r')
        meter.role('p', 'decode')
        now[0] = 3.
        window = meter.snapshot(reset=True)
        self.assertAlmostEqual(window['decode']['utilization'], 2/4)
        self.assertAlmostEqual(window['decode']['mean_queue'], 2/3)
        self.assertEqual(window['decode']['queue_depth'], 0)
        self.assertEqual(meter.snapshot()['decode']['utilization'], 0)

    def test_only_sustained_long_interval_pressure_moves_roles(self):
        profile = Profile((1,100), (.1,.1), (100,100))
        control = Controller(Workload(3,.5,1,100),profile,profile,2,4,100000,1,1,1,
                             [0,10,20,50,100], long_interval=5)
        first = control.update(0,100000,0,0)
        stages = {stage: dict(mean_queue=0., utilization=.1, instances=1) for stage in StageLoad.stages}
        stages['remote_prefill'].update(mean_queue=2., utilization=.99)
        stages['local_prefill']['instances'] = first['prefill_instances']
        self.assertEqual(control.update(2,100000,0,0,stages=stages)['prefill_instances'], first['prefill_instances'])
        second = control.update(6,100000,0,0,stages=stages)
        self.assertEqual(second['stage_adjustment'], 'prefill_queue')
        self.assertEqual(second['prefill_instances'], first['prefill_instances'] + 1)
        stages['remote_prefill'].update(mean_queue=0., utilization=.5)
        stages['local_prefill']['instances'] = second['prefill_instances']
        stages['decode'].update(mean_queue=2., utilization=.95)
        third = control.update(12,100000,0,0,stages=stages)
        self.assertEqual(third['stage_adjustment'], 'decode_queue')
        self.assertEqual(third['prefill_instances'], second['prefill_instances'] - 1)
