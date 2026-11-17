import unittest
from prfaas.cache import HybridCache, CacheDirectory
from prfaas.scheduler import Router, Controller
from prfaas.analytics import Profile, Workload

class ControlTests(unittest.TestCase):
    def test_cache_requires_exact_recurrent_checkpoint(self):
        c=HybridCache(block_tokens=2,block_bytes=16,capacity_blocks=20,namespace='model-a')
        c.put((1,2,3,4),{'full':(b'ab',b'cd')},{'linear':b'state4'})
        self.assertEqual(c.match((1,2,9,9)).length,0)
        c.put((1,2),{'full':(b'ab',)},{'linear':b'state2'})
        m=c.match((1,2,9,9))
        self.assertEqual(m.length,2)
        self.assertEqual(m.recurrent['linear'],b'state2')
        self.assertEqual(c.match((1,2,3,4,5)).length,4)
        with self.assertRaises(ValueError): c.put((1,2,3),{'full':(b'ab',b'c')},{'linear':b'bad'})

    def test_pinned_transfer_shared_capacity_and_cleanup(self):
        c=HybridCache(2,4,2,'model')
        handle=c.stage_transfer(b'12345678')
        self.assertEqual(c.used_blocks,2)
        with self.assertRaises(MemoryError): c.stage_transfer(b'1')
        with self.assertRaises(ValueError): c.complete_transfer(handle,acknowledged=False)
        self.assertEqual(c.used_blocks,2)
        c.complete_transfer(handle,acknowledged=True)
        self.assertEqual(c.used_blocks,0)

    def test_lru_and_namespace(self):
        c=HybridCache(2,8,2,'model')
        c.put((1,2),{'f':(b'12',)},{'r':b'2'})
        c.put((3,4),{'f':(b'34',)},{'r':b'4'})
        self.assertEqual(c.match((1,2)).length,0)
        self.assertEqual(c.match((3,4)).length,2)
        d=CacheDirectory(); d.register('local','node1',c)
        self.assertEqual(d.best('local','other',(3,4)),(None,0))
        self.assertEqual(d.best('local','model',(3,4)),('node1',2))

    def test_routing_bandwidth_and_cache_branches(self):
        r=Router(threshold=4)
        self.assertEqual(r.route(10,6,0,False).target,'local')
        self.assertEqual(r.route(10,0,8,False).target,'remote')
        a=r.route(10,0,8,True)
        self.assertEqual((a.target,a.prefix_source,a.prefix_tokens),('local','remote',8))
        b=r.route(10,8,0,True)
        self.assertEqual((b.target,b.prefix_source),('local','local'))
        with self.assertRaises(ValueError): r.route(10,11,0,False)
        self.assertEqual(Router(0,remote_available=False).route(10,0,0,True).target,'local')

    def test_controller_short_and_long_timescales(self):
        p=Profile((1.,100.),(1.,10.),(100.,1000.))
        control=Controller(Workload(3,.5,1,100),p,p,2,4,1000,32,.025,1024,[0,10,20,50,100],long_interval=60)
        first=control.update(now=0,bandwidth_bytes_s=1000,utilization=.1,queue_depth=0)
        old_count=first['prefill_instances']
        second=control.update(now=1,bandwidth_bytes_s=10,utilization=.99,queue_depth=20)
        self.assertEqual(second['prefill_instances'],old_count)
        self.assertFalse(control.bandwidth_abundant)
        third=control.update(now=61,bandwidth_bytes_s=1000,utilization=.1,queue_depth=0)
        self.assertIsNotNone(third)
        self.assertTrue(control.bandwidth_abundant)
        self.assertEqual(third['prefill_instances']+third['decode_instances'],4)
        control.update(now=62,bandwidth_bytes_s=0,utilization=0,queue_depth=0)
        self.assertFalse(control.bandwidth_abundant)
        self.assertFalse(control.router.route(10,0,8,control.bandwidth_abundant).migrate_prefix)
        updated=Workload(2,.5,1,100)
        control.update(now=122,bandwidth_bytes_s=1000,utilization=0,queue_depth=0,workload=updated)
        self.assertIs(control.args[0],updated)

if __name__=='__main__': unittest.main()
