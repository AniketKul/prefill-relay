"""Single-host, independent-process deployment of selective prefill serving."""
from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
import secrets
import threading
import time
from uuid import uuid4

from .analytics import Profile, count, positive
from .empirical import EmpiricalWorkload
from .scheduler import Controller, Router
from .telemetry import LinkMonitor, RolePool
from .worker import WorkerProcess


class Coordinator:
    """Logical clusters share one machine; rates are application-level emulation.

    A shared reservation clock budgets each cross-cluster frame and migration,
    while independent producers compute concurrently.
    """
    def __init__(self,path,backend='tiny',device='cpu',model_id=None,revision='main',
                 bandwidth_bytes_s=100*2**20,threshold=128,profile=None,output_tokens=8,
                 local_workers=2,remote_workers=1,connections=4,cache_bytes=4*2**30,max_batch=4,batch_wait_s=.005,receive_bytes=4*2**30,block_tokens=16,
                 local_rdma_address=None, decode_slo_tps=None):
        count(local_workers,'local workers',zero=False); count(remote_workers,'remote workers',zero=False)
        for value,name in [(connections,'connections'),(max_batch,'maximum batch'),(output_tokens,'output tokens'),
                           (cache_bytes,'cache bytes'),(receive_bytes,'receive bytes'),(block_tokens,'block tokens')]:
            count(value,name,zero=False)
        if cache_bytes<4096 or max_batch>8 or connections>64: raise ValueError('cache >=4096 bytes, batch <=8 and connections <=64 required')
        positive(threshold,'threshold',zero=True)
        if local_workers<2: raise ValueError('at least two local workers required')
        self.path=Path(path).expanduser().resolve(); self.path.mkdir(parents=True,exist_ok=True)
        self.decode_calibration = None
        if decode_slo_tps is not None:
            if profile is None: raise ValueError('decode SLO calibration requires a measured profile')
            from .calibration import select_decode_batch
            measured = profile.get('measurements', profile)
            self.decode_calibration = select_decode_batch(measured['rows'], measured.get('decode_context_lengths', measured['profile']['lengths']),
                                                         decode_slo_tps, max_batch)
            max_batch = self.decode_calibration['batch_max']
            if not max_batch: raise ValueError('no measured batch meets the requested decode SLO')
        self.config=dict(local_rdma_address=local_rdma_address,backend=backend,device=device,model_id=model_id,revision=revision,
                         connections=connections,block_tokens=block_tokens,max_batch=max_batch,batch_wait_s=batch_wait_s,cache_bytes=cache_bytes,receive_bytes=receive_bytes,token=secrets.token_urlsafe(32))
        self.nlocal=local_workers; self.nremote=remote_workers; self.threshold=threshold
        self.monitor=LinkMonitor(bandwidth_bytes_s); self.bandwidth=bandwidth_bytes_s
        self.control_lock=threading.RLock(); self.egress=threading.Lock(); self.queue=0
        self.observations=deque(maxlen=256); self.controller=None; self.profile=profile
        self.replications = set()
        self.output_tokens=output_tokens; self.workers={}; self.locks={}; self.receipts=[]

    def __enter__(self):
        from .cache_service import CacheClient
        from .directory import GlobalDirectory
        from .rate_limit import RateService
        self.stack=ExitStack(); self.directory=GlobalDirectory()
        self.rate=self.stack.enter_context(RateService(self.bandwidth,self.config['token']))
        self.config['rate_url']=self.rate.url
        try:
            for cluster,n in [('remote',self.nremote),('local',self.nlocal)]:
                for i in range(n):
                    name=f'{cluster}-{i}'; role='decode' if cluster=='local' and i else 'prefill'
                    config={**self.config,'local_rdma_address':self.config['local_rdma_address'] if cluster=='local' else None,'role':role,'cache_path':str(self.path/(name+'.sqlite'))}
                    worker=self.stack.enter_context(WorkerProcess(config)); self.workers[name]=worker
                    self.locks[name]=threading.Lock(); info=worker.info
                    self.directory.register(cluster,name,CacheClient(info['cache_url'],info['namespace'],self.config['token'],timeout=120,max_body_bytes=self.config['cache_bytes']*2))
            namespaces={w.info['namespace'] for w in self.workers.values()}
            if len(namespaces)!=1: raise ValueError('worker model identities differ')
            self.namespace=namespaces.pop()
            self.roles=RolePool({n:w.info['role'] for n,w in self.workers.items() if n.startswith('local')},
                                lambda n,r:self.workers[n].call('role',role=r))
            if self.profile is not None:
                raw=self.profile['measurements'] if 'measurements' in self.profile else self.profile
                first=next(iter(self.workers.values())).info
                if raw.get('model',first['model_id'])!=first['model_id'] or raw.get('revision',first['revision'])!=first['revision']:
                    raise ValueError('profile model or revision differs from workers')
                measured=Profile(**raw['profile']); self.measured=measured
                rows=raw['rows']; batch=1
                step=max(r['decode_step_seconds'] for r in rows if r.get('batch',1)==batch)
                if self.decode_calibration:
                    batch = self.decode_calibration['batch_max']
                    step = self.decode_calibration['decode_step_s']
                self.controller=Controller(EmpiricalWorkload([measured.lengths[0]],low=measured.lengths[0],high=measured.lengths[-1]),
                    measured,measured,self.nremote,self.nlocal,self.bandwidth,batch,step,self.output_tokens,
                    [0,*measured.lengths],long_interval=5)
                self.controller.update(time.monotonic(),self.bandwidth,0,0)
            return self
        except BaseException:
            self.stack.close(); raise

    def __exit__(self,*exc): self.stack.close()

    def set_bandwidth(self,bytes_s):
        positive(bytes_s,'bandwidth',zero=True)
        with self.control_lock:
            self.bandwidth=bytes_s; self.monitor.capacity=bytes_s
            if hasattr(self,'rate'): self.rate.set_bandwidth(bytes_s)

    def reset_caches(self):
        with self.control_lock:
            if any(lock.locked() for lock in self.locks.values()) or any(self.roles.active.values()): raise RuntimeError('requests still active')
            for worker in self.workers.values(): worker.call('clear')
            self.observations.clear(); self.monitor.events.clear()
            if self.controller:
                self.controller.current=None; self.controller.last_long=None; self.controller.last_time=None

    def _update_loads(self):
        for name in self.workers:
            cluster='local' if name.startswith('local') else 'remote'
            load=self.roles.active.get(name,int(self.locks[name].locked()))
            self.directory.update_load(cluster,name,load)

    def _choose(self,tokens,policy):
        self._update_loads()
        local=self.directory.best('local',self.namespace,tokens)
        remote=self.directory.best('remote',self.namespace,tokens)
        egress_queue=self.rate.stats()['pending_reservations']
        in_flight = [worker.call('link') for name, worker in self.workers.items()
                     if name.startswith('remote') and self.locks[name].locked()]
        stats=self.monitor.snapshot(egress_queue, in_flight=in_flight)
        if policy=='all-local':
            self.roles.reallocate(1); router=Router(len(tokens),False,True); abundant=False
        elif policy=='all-remote':
            if not self.bandwidth: raise RuntimeError('remote link unavailable')
            self.roles.reallocate(0); router=Router(0,True,False); abundant=False
        else:
            abundant=self.bandwidth>0 and stats['utilization']<.85 and egress_queue<8
            uncached=len(tokens)-(max(local[1],remote[1]) if abundant else local[1])
            if self.controller:
                samples=EmpiricalWorkload(self.observations or [uncached],low=self.measured.lengths[0],high=self.measured.lengths[-1])
                current=self.controller.update(time.monotonic(),stats['bandwidth_bytes_s'],stats['utilization'],egress_queue,
                                               workload=samples if samples.has_misses else None)
                self.roles.reallocate(current['prefill_instances']); router=self.controller.router
                abundant=self.controller.bandwidth_abundant
            else:
                self.roles.reallocate(1); router=Router(self.threshold,self.bandwidth>0,True)
        placement=router.route(len(tokens),local[1],remote[1],abundant)
        return placement,{'local':local,'remote':remote},stats

    def run(self,tokens,output_tokens=8,policy='selective'):
        if policy not in ('selective','all-local','all-remote'): raise ValueError('invalid policy')
        if not isinstance(tokens,list) or not tokens or any(type(t) is not int or t<0 for t in tokens): raise ValueError('invalid tokens')
        count(output_tokens,'output tokens',zero=False)
        start=time.perf_counter(); request=uuid4().hex; acquired=[]; roles=ExitStack(); production=ExitStack()
        cached=self._cached(tokens,output_tokens,start,request)
        if cached is not None: return cached
        with self.control_lock: self.queue+=1
        try:
            while True:
                with self.control_lock:
                    placement,hits,telemetry=self._choose(tokens,policy)
                    decoders=self.roles.available('decode')
                    producers=self.roles.available('prefill') if placement.target=='local' else [n for n in self.workers if n.startswith('remote')]
                    decoders.sort(key=lambda n:(-self.directory._client(('local',n),self.namespace).match_length(tokens),n))
                    producers.sort(key=lambda n:(n!=hits[placement.target][0],n))
                    pair=next(((p,d) for p in producers for d in decoders if not self.locks[p].locked() and self.roles.active[d]<self.config['max_batch']),None)
                    if pair:
                        producer,decoder=pair
                        self.locks[producer].acquire(); acquired.append(producer)
                        roles.enter_context(self.roles.acquire(decoder,'decode'))
                        if placement.target=='local': production.enter_context(self.roles.acquire(producer,'prefill'))
                        break
                time.sleep(.01)
            with roles:
                roles.callback(self._discard,decoder,request)
                destination=self.workers[decoder].call('prepare',request=request,tokens=tokens)
                cross=placement.target=='remote' or placement.migrate_prefix
                bandwidth=self.bandwidth
                if cross and not bandwidth: raise RuntimeError('remote link unavailable')
                migration=None; moved=0
                source_node=hits[placement.prefix_source][0]
                if placement.prefix_tokens and source_node and source_node!=producer:
                    before=time.perf_counter()
                    migration=self.directory.migrate((placement.prefix_source,source_node),(placement.target,producer),
                        self.namespace,tokens,budget_bytes=self.config['cache_bytes']//2)
                    if placement.prefix_source!=placement.target:
                        moved=migration.bytes_transferred
                        migration_elapsed=time.perf_counter()-before
                    self.receipts.append(asdict(migration))
                rdma = bool(self.config['local_rdma_address'] and placement.target=='local')
                endpoint = 'rdma_receiver' if rdma else 'receiver'
                pref=self.workers[producer].call('prefill',request=request,tokens=tokens,
                    fingerprints=destination['fingerprints'],rdma=rdma,destination=self.workers[decoder].info[endpoint],destination_prefix=destination['prefix_tokens'],
                    rate_limit=placement.target=='remote')
                with self.control_lock: self.observations.append(len(tokens)-pref['prefix_tokens'])
                if placement.target=='remote':
                    moved+=pref['sender']['bytes']; self.monitor.record(pref['sender'])
                if migration and migration.bytes_transferred and placement.prefix_source!=placement.target:
                    self.monitor.record({'bytes':migration.bytes_transferred,'elapsed_s':migration_elapsed})
                with self.control_lock:
                    production.close()
                    self.locks[producer].release(); acquired.remove(producer)
                self.workers[decoder].call('consume',rdma=rdma,request=request,first_token=pref['first_token'])
                ttft=time.perf_counter()-start
                decoded=self.workers[decoder].call('decode',request=request,output_tokens=output_tokens)
                return dict(ttft_s=ttft,total_s=time.perf_counter()-start,decode_s=decoded['decode_s']+decoded['batch_queue_s'],
                    batch_size=decoded['batch_size'],batch_queue_s=decoded['batch_queue_s'],
                    generated_tokens=decoded['generated_tokens'],transferred_bytes=moved,target=placement.target,
                    prefix_tokens=pref['prefix_tokens'],destination_prefix_tokens=destination['prefix_tokens'],
                    migration=asdict(migration) if migration else None,handoff_transport='rdma' if rdma else 'tcp',prefill=pref,telemetry=telemetry,
                    producer=producer,decoder=decoder,worker_pids={n:self.workers[n].info['pid'] for n in (producer,decoder)})
        except Exception as error:
            if getattr(error,'transport_stats',None): self.monitor.record(error.transport_stats)
            raise
        finally:
            production.close()
            for n in acquired:
                try: self.workers[n].call('discard',request=request)
                except Exception: pass
            with self.control_lock:
                for n in acquired: self.locks[n].release()
                self.queue-=1

    def _cached(self,tokens,output_tokens,start,request):
        reservation=ExitStack()
        with self.control_lock:
            nodes=[n for n in self.roles.available('decode') if self.roles.active[n]<self.config['max_batch']]
            node=next((n for n in nodes if self.directory._client(('local',n),self.namespace).match_length(tokens)==len(tokens)),None)
            replica = None
            if node is None and nodes:
                # A hot decoder has no free slots. Replicate its checkpoint onto an idle decoder.
                saturated = [n for n in self.roles.available('decode')
                             if self.roles.active[n] >= self.config['max_batch']]
                source = next((n for n in saturated if
                               self.directory._client(('local',n),self.namespace).match_length(tokens)==len(tokens)), None)
                node = next((n for n in nodes if self.roles.active[n]==0 and
                             n not in self.replications), None) if source else None
                if node:
                    replica = ('local', source)
                    self.replications.add(node)
                    reservation.callback(self._finish_replication, node)
            if node is None: return None
            reservation.enter_context(self.roles.acquire(node,'decode'))
        with reservation:
            if replica:
                try:
                    copy = self.migrate_prefix(replica, ('local',node), tokens)
                except (ConnectionError, MemoryError):
                    return None
                if copy['length'] != len(tokens) or copy['status'] not in ('already_cached','transferred'):
                    return None
            reservation.callback(self._discard,node,request)
            prepared=self.workers[node].call('prepare',request=request,tokens=tokens,cached_only=True)
            if prepared['prefix_tokens']!=len(tokens): return None
            self.workers[node].call('consume_cached',request=request,length=len(tokens))
            ttft=time.perf_counter()-start
            decoded=self.workers[node].call('decode',request=request,output_tokens=output_tokens)
            with self.control_lock: self.observations.append(0)
            return dict(ttft_s=ttft,total_s=time.perf_counter()-start,
                decode_s=decoded['decode_s']+decoded['batch_queue_s'],batch_size=decoded['batch_size'],
                batch_queue_s=decoded['batch_queue_s'],generated_tokens=decoded['generated_tokens'],
                transferred_bytes=0,target='local',prefix_tokens=len(tokens),destination_prefix_tokens=len(tokens),
                migration=copy if replica else None,handoff_transport='cache',producer=None,decoder=node,worker_pids={node:self.workers[node].info['pid']},
                prefill={'cache_hit':True,'prefill_s':0,'sender':{'bytes':0},'pipelined':False,'layer_ready_s':[]})

    def _finish_replication(self, node):
        with self.control_lock:
            self.replications.discard(node)

    def _discard(self,node,request):
        try: self.workers[node].call('discard',request=request)
        except Exception: pass

    def rebalance(self,tokens,budget_bytes=None,allow_cross_cluster=False):
        with self.control_lock: self._update_loads()
        budget=self.config['cache_bytes']//2 if budget_bytes is None else budget_bytes
        with self.egress:
            start=time.perf_counter()
            if allow_cross_cluster and not self.bandwidth: raise RuntimeError('remote link unavailable')
            receipt=self.directory.rebalance(self.namespace,tokens,budget_bytes=budget,allow_cross_cluster=allow_cross_cluster)
            if receipt.source[0]!=receipt.target[0] and receipt.bytes_transferred:
                self.monitor.record({'bytes':receipt.bytes_transferred,'elapsed_s':time.perf_counter()-start})
            self.receipts.append(asdict(receipt)); return asdict(receipt)

    def migrate_prefix(self,source,target,tokens,budget_bytes=None):
        start=time.perf_counter()
        budget=self.config['cache_bytes']//2 if budget_bytes is None else budget_bytes
        receipt=self.directory.migrate(source,target,self.namespace,tokens,budget)
        if source[0]!=target[0] and receipt.bytes_transferred:
            self.monitor.record({'bytes':receipt.bytes_transferred,'elapsed_s':time.perf_counter()-start})
        self.receipts.append(asdict(receipt)); return asdict(receipt)

    def warmup(self,prompts):
        for node,worker in self.workers.items():
            with self.locks[node]:
                for tokens in prompts: worker.call('warmup',tokens=tokens)
        self.reset_caches()

    def reference(self,tokens,output_tokens=8):
        node=next(iter(self.workers))
        with self.locks[node]: return self.workers[node].call('reference',tokens=tokens,output_tokens=output_tokens)

    def stats(self):
        return {'decode_calibration':self.decode_calibration,'workers':{n:w.call('stats') for n,w in self.workers.items()},'roles':dict(self.roles.roles),
                'role_history':self.roles.history,'telemetry':self.monitor.snapshot(self.rate.stats()['pending_reservations']),'rate':self.rate.stats(),
                'outstanding_requests':self.queue,'migrations':self.receipts}
