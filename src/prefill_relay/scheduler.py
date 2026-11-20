"""Bandwidth/cache-aware routing and dual-timescale control."""
from dataclasses import dataclass
from .analytics import count, positive, grid_search


@dataclass(frozen=True)
class Placement:
    target: str
    prefix_source: str
    prefix_tokens: int
    uncached_tokens: int
    migrate_prefix: bool


class Router:
    def __init__(self,threshold,remote_available=True,local_available=True):
        positive(threshold,'threshold',zero=True)
        if not remote_available and not local_available: raise ValueError('no prefill capacity')
        self.threshold=threshold; self.remote_available=remote_available; self.local_available=local_available

    def route(self,total,local_cached,remote_cached,bandwidth_abundant):
        count(total,'total',zero=False)
        for n in (local_cached,remote_cached):
            count(n,'prefix')
            if n>total: raise ValueError('prefix exceeds prompt')
        best=max(local_cached,remote_cached) if bandwidth_abundant else local_cached
        target='local' if total-best<=self.threshold else 'remote'
        if not self.remote_available: target='local'
        if not self.local_available: target='remote'
        cached={'local':local_cached,'remote':remote_cached}
        source=target
        if bandwidth_abundant and cached['remote' if target=='local' else 'local']>cached[target]:
            source='remote' if target=='local' else 'local'
        prefix=cached[source]
        return Placement(target,source,prefix,total-prefix,source!=target and prefix>0)


class Controller:
    """Configurable telemetry triggers for threshold and allocation updates."""
    def __init__(self,workload,remote_profile,local_profile,n_remote,n_local,bandwidth_bytes_s,batch_max,decode_step_s,output_tokens,thresholds,long_interval=60,utilization_trigger=.85,queue_trigger=8):
        positive(long_interval,'long interval'); positive(utilization_trigger,'utilization trigger')
        if utilization_trigger>1: raise ValueError('utilization trigger must be <=1')
        count(queue_trigger,'queue trigger',zero=False)
        self.args=(workload,remote_profile,local_profile,n_remote,n_local)
        self.decode=(batch_max,decode_step_s,output_tokens)
        self.thresholds=tuple(thresholds); self.long_interval=long_interval
        self.utilization_trigger=utilization_trigger; self.queue_trigger=queue_trigger
        self.last_long=None; self.last_time=None; self.current=None; self.bandwidth_abundant=True
        self.bandwidth=bandwidth_bytes_s

    def update(self,now,bandwidth_bytes_s,utilization,queue_depth,workload=None,stages=None):
        positive(now,'time',zero=True); positive(bandwidth_bytes_s,'bandwidth',zero=True)
        positive(utilization,'utilization',zero=True); count(queue_depth,'queue depth')
        if utilization>1: raise ValueError('utilization must be <=1')
        if self.last_time is not None and now<self.last_time: raise ValueError('time cannot move backwards')
        self.last_time=now
        abundant=bandwidth_bytes_s>0 and utilization<self.utilization_trigger and queue_depth<self.queue_trigger
        long_due=self.last_long is None or now-self.last_long>=self.long_interval
        if self.current and bandwidth_bytes_s==0 and self.current['prefill_instances']==0 and self.args[4]>1:
            long_due=True
        changed=bandwidth_bytes_s!=self.bandwidth or abundant!=self.bandwidth_abundant
        if workload is not None:
            self.args=(workload,*self.args[1:])
            changed=True
        self.bandwidth_abundant=abundant; self.bandwidth=bandwidth_bytes_s
        if long_due or changed or not abundant:
            counts=None if long_due else [self.current['prefill_instances']]
            adjustment = None
            if long_due and self.current and stages and bandwidth_bytes_s:
                decode = stages['decode']
                prefill = [stages['local_prefill'], stages['remote_prefill']]
                def congested(stage):
                    return stage['mean_queue'] >= .5 and stage['utilization'] >= self.utilization_trigger
                active = stages['local_prefill']['instances']
                if congested(decode) and active and not any(congested(s) for s in prefill):
                    counts, adjustment = [active - 1], 'decode_queue'
                elif (any(congested(s) for s in prefill) and decode['utilization'] < self.utilization_trigger
                      and active + 1 < self.args[4]):
                    counts, adjustment = [active + 1], 'prefill_queue'
            # Target utilization leaves capacity for queued frames to drain.
            budget = bandwidth_bytes_s
            if utilization >= self.utilization_trigger:
                budget *= self.utilization_trigger
            if queue_depth >= self.queue_trigger:
                budget *= self.queue_trigger / (queue_depth + 1)
            def search(counts):
                best, _ = grid_search(*self.args, budget, *self.decode, self.thresholds,
                                      prefill_counts=counts)
                return best
            best = search(counts)
            if best['throughput'] == 0 and counts is not None:
                best = search(None)
                adjustment = None
                long_due = True
            if best['throughput'] == 0:
                self.current = None
                self.last_long = None
                raise ValueError('no feasible prefill and decode allocation on this threshold grid')
            self.current = dict(best, routing_bandwidth_bytes_s=budget)
            if adjustment:
                self.current['stage_adjustment'] = adjustment
            if long_due: self.last_long=now
        return self.current

    @property
    def router(self):
        if self.current is None: raise RuntimeError('initialize controller with telemetry first')
        return Router(self.current['threshold'],self.args[3]>0 and self.bandwidth>0,self.current['prefill_instances']>0)
