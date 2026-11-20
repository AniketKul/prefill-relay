import math
import unittest
from prefill_relay.analytics import Profile, Workload, capacity, grid_search, kv_rate, required_egress

class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.remote = Profile((1., 100.), (2., 2.), (100., 100.))
        self.local = Profile((1., 100.), (4., 4.), (100., 100.))

    def test_equations_and_aggregate_bandwidth(self):
        r = capacity(.5, 75, 25, self.remote, self.local, 4, 4, 2, 100, 32, .025, 1024)
        self.assertEqual(r.remote, 1)
        self.assertEqual(r.local, 1)
        self.assertEqual(r.decode, 2.5)
        self.assertEqual(r.total, 2)
        self.assertEqual(kv_rate(100, 2), 50)
        self.assertEqual(required_egress(32, 8, [100, 300], [1, 3]), 400)

    def test_complete_hits_still_consume_decode_capacity(self):
        from prefill_relay.empirical import EmpiricalWorkload
        # Each remote miss transfers 100 bytes through a 100-byte/s link.
        def estimate(samples, batch):
            row, _ = grid_search(EmpiricalWorkload(samples), self.remote, self.local,
                                 4, 1, 100, batch, 1, 1, [0])
            return row['throughput'], row['egress_gbps'] * 1e9 / 8
        self.assertEqual(estimate([50, 50], 4), (1, 100))
        self.assertEqual(estimate([0, 50], 4), (2, 100))
        self.assertEqual(estimate([0, 50], 1), (1, 50))

    def test_unused_stages_do_not_limit_capacity(self):
        a = capacity(0, None, 25, self.remote, self.local, 0, 4, 2, 0, 32, .025, 1024)
        b = capacity(1, 75, None, self.remote, self.local, 4, 0, 2, 1000, 32, .025, 1024)
        self.assertEqual(a.total, 1)
        self.assertEqual(b.total, 2)
        self.assertEqual(capacity(.5, 75, 25, self.remote, self.local, 0, 4, 2, 1000,32,.025,1024).total,0)

    def test_truncated_distribution(self):
        w=Workload()
        p,long,short=w.split(19400)
        self.assertAlmostEqual(p,.49486855,places=6)
        self.assertAlmostEqual(w.mean,27312.73,places=1)
        self.assertAlmostEqual(p*long+(1-p)*short,w.mean,places=6)
        self.assertEqual(w.split(0)[0],1)
        self.assertEqual(w.split(128000)[0],0)
        self.assertEqual(w.split(math.nextafter(128000,128))[0],0)
        self.assertEqual(w.split(math.nextafter(128,128000))[0],1)
        values=w.sample(1000,seed=7)
        self.assertTrue(all(128<v<128000 for v in values))
        self.assertEqual(values,w.sample(1000,seed=7))

    def test_search_is_global_best_on_supplied_grid(self):
        w=Workload(mu=3,sigma=.5,low=1,high=100)
        best, rows=grid_search(w,self.remote,self.local,4,6,1000,32,.025,1024,[0,10,20,50,100])
        self.assertEqual(best['throughput'],max(r['throughput'] for r in rows))
        self.assertEqual(len(rows),30)
        self.assertEqual(best['prefill_instances']+best['decode_instances'],6)

    def test_invalid_values_rejected(self):
        for v in [0,-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError): Profile((1.,2.),(v,2.),(1.,2.))
        with self.assertRaises(ValueError): self.remote.at(101)
        with self.assertRaises(ValueError): Workload(sigma=0)
        with self.assertRaises(ValueError): kv_rate(1,0)
        with self.assertRaises(ValueError): grid_search(Workload(),self.remote,self.local,4,0,1,1,1,1,[1])

if __name__=='__main__': unittest.main()
