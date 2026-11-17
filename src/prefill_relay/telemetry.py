"""Live link feedback and drain-safe role allocation for homogeneous workers."""
from collections import deque
from contextlib import contextmanager
from threading import RLock
import time
from .analytics import count, positive


class LinkMonitor:
    def __init__(self,capacity_bytes_s,window_s=5):
        positive(capacity_bytes_s,'link capacity',zero=True); positive(window_s,'window')
        self.capacity=capacity_bytes_s; self.window=window_s; self.events=deque(); self.lock=RLock()

    def record(self,stats):
        with self.lock:
            self.events.append((time.monotonic(),dict(stats)))

    def snapshot(self,queue_depth=0):
        count(queue_depth,'queue depth'); now=time.monotonic()
        with self.lock:
            while self.events and now-self.events[0][0]>self.window: self.events.popleft()
            rows=[s for _,s in self.events]
            volume=sum(s.get('bytes',0) for s in rows)
            elapsed=sum(s.get('elapsed_s',0) for s in rows)
            failures=sum(s.get('failures',0) for s in rows)
            retrans=sum(s.get('retransmissions',0) for s in rows)
            observed=volume/elapsed if elapsed>0 else self.capacity
            span=max([now-self.events[0][0] if self.events else 0,*[s.get('elapsed_s',0) for s in rows],1e-6])
            utilization=min(1.,volume/span/self.capacity) if self.capacity else 1.
            effective=min(self.capacity,observed/(1+failures+retrans)) if failures or retrans else self.capacity
            return dict(bandwidth_bytes_s=effective,utilization=utilization,queue_depth=queue_depth,
                        goodput_bytes_s=observed,failures=failures,retransmissions=retrans,
                        rtt_ms=max((s.get('rtt_ms',0) for s in rows),default=0))


class RolePool:
    def __init__(self,roles,apply_role):
        if not roles or any(r not in ('prefill','decode') for r in roles.values()) or 'decode' not in roles.values():
            raise ValueError('local workers require valid roles and a decode worker')
        self.roles=dict(roles); self.active={n:0 for n in roles}; self.pending={}; self.apply_role=apply_role
        self.lock=RLock(); self.history=[]

    @contextmanager
    def acquire(self,node,role):
        with self.lock:
            if node in self.pending or self.roles.get(node)!=role: raise RuntimeError('worker draining or wrong role')
            self.active[node]+=1
        try: yield
        finally:
            with self.lock:
                self.active[node]-=1
                self._apply_ready()

    def _apply_ready(self):
        # Convert idle workers together only when at least one decoder remains available.
        for node,role in list(self.pending.items()):
            if self.active[node]: continue
            if role=='prefill' and sum(r=='decode' for r in self.roles.values())==1: continue
            self.apply_role(node,role)
            self.roles[node]=role; del self.pending[node]
            self.history.append({'node':node,'role':role,'at':time.monotonic()})

    def reallocate(self,n_prefill):
        count(n_prefill,'prefill count')
        if n_prefill>=len(self.roles): raise ValueError('a decoder must remain')
        with self.lock:
            desired={n:('prefill' if i<n_prefill else 'decode') for i,n in enumerate(sorted(self.roles))}
            self.pending={n:r for n,r in desired.items() if self.roles[n]!=r}
            self._apply_ready(); self._apply_ready()

    def available(self,role):
        with self.lock:
            return sorted((n for n,r in self.roles.items() if r==role and n not in self.pending),key=lambda n:self.active[n])
