import threading
import time
import unittest
from prefill_relay.empirical import EmpiricalWorkload
from prefill_relay.workload import arrival_offsets, compare_policies, run_workload, skew_prompts


class FakeCoordinator:
    def __init__(self, delay=.01):
        self.delay=delay; self.calls=[]; self.resets=0; self.bandwidth=[]
        self.lock=threading.Lock()

    def reset_caches(self): self.resets+=1
    def set_bandwidth(self, value): self.bandwidth.append(value)

    def run(self, tokens, output_tokens, policy='selective'):
        with self.lock: self.calls.append((tuple(tokens), output_tokens, policy))
        time.sleep(self.delay)
        return dict(ttft_s=.001, total_s=self.delay, decode_s=.003,
                    generated_tokens=[7]*output_tokens, transferred_bytes=100,
                    target='local' if policy=='all-local' else 'remote', prefix_tokens=0)


class WorkloadTests(unittest.TestCase):
    def test_arrivals_and_skew_are_deterministic(self):
        self.assertEqual(arrival_offsets(4,2), [0,.5,1,1.5])
        self.assertEqual(arrival_offsets(6,2,'burst',3), [0,0,0,1.5,1.5,1.5])
        a=skew_prompts([[1],[2]],100,seed=9)
        self.assertEqual(a,skew_prompts([[1],[2]],100,seed=9))
        self.assertGreater(a.count([1]),80)
        with self.assertRaises(ValueError): arrival_offsets(1,0)
        with self.assertRaises(ValueError): skew_prompts([[1]],2,1.1)

    def test_queue_time_is_part_of_ttft_and_goodput(self):
        result=run_workload(FakeCoordinator(),[[1],[2],[3]], output_tokens=4,
                            rate=1e6,max_workers=1,max_pending=3,ttft_slo_s=.005)
        rows=result['requests']; summary=result['summary']
        self.assertEqual(summary['completed'],3)
        self.assertGreater(rows[-1]['queue_s'],.01)
        self.assertAlmostEqual(rows[-1]['ttft_s'],rows[-1]['queue_s']+.001)
        self.assertEqual(summary['mean_decode_tps'],1000)
        self.assertLessEqual(summary['slo_passed'],1)
        self.assertEqual(summary['transferred_bytes'],300)
        self.assertGreater(summary['p90_ttft_s'],summary['mean_ttft_s'])

    def test_bounded_overload_and_errors(self):
        result=run_workload(FakeCoordinator(.04),[[1],[2],[3]],rate=1e6,
                            max_workers=1,max_pending=1)
        self.assertEqual(result['summary']['rejected'],2)
        self.assertEqual(result['summary']['completed'],1)
        class Broken(FakeCoordinator):
            def run(self,*args,**kwargs): raise RuntimeError('worker unavailable')
        broken=run_workload(Broken(),[[1]])
        self.assertEqual(broken['summary']['errors'],1)
        self.assertEqual(broken['summary']['slo_goodput_rps'],0)
        self.assertIn('worker unavailable',broken['requests'][0]['error'])

    def test_comparison_reuses_inputs_and_resets_warmup(self):
        fake=FakeCoordinator(.001)
        results=compare_policies(fake,[[1],[1,2]],output_tokens=2,
                                 policies=('all-local','all-remote'),warm_rounds=1,
                                 rate=100,initial_bandwidth_bytes_s=1000,
                                 bandwidth_changes=((.005,100),))
        self.assertEqual(fake.resets,2)
        self.assertEqual(fake.bandwidth,[1000,100,1000,100])
        self.assertEqual(len(fake.calls),8)
        self.assertEqual([c[:2] for c in fake.calls[:4]], [c[:2] for c in fake.calls[4:]])
        self.assertEqual(set(results),{'all-local','all-remote'})
        self.assertEqual(results['all-local']['summary']['requests'],2)
        with self.assertRaises(ValueError):
            run_workload(fake,[[1]],bandwidth_changes=((0,0),))

    def test_single_output_has_no_decode_interval(self):
        result=run_workload(FakeCoordinator(.001),[[1]],output_tokens=1)
        self.assertIsNone(result['requests'][0]['decode_tps'])
        self.assertEqual(result['summary']['slo_passed'],1)


class EmpiricalTests(unittest.TestCase):
    def test_cache_hits_are_separate_from_miss_distribution(self):
        workload=EmpiricalWorkload([0,2,10,10],low=1,high=10)
        self.assertEqual(workload.full_hit_fraction,.25)
        self.assertEqual(workload.split(2),(2/3,10,2))
        self.assertAlmostEqual(workload.mean,22/3)
        self.assertEqual(workload.split(10),(0,None,22/3))

    def test_all_hits_and_outside_profile_are_explicit(self):
        workload=EmpiricalWorkload([0,0])
        self.assertFalse(workload.has_misses)
        self.assertEqual(workload.full_hit_fraction,1)
        with self.assertRaises(ValueError): workload.split(10)
        for lengths in ([],[-1],[True],[101],[.5]):
            with self.assertRaises(ValueError): EmpiricalWorkload(lengths,low=1,high=100)


if __name__=='__main__': unittest.main()
