"""Prefill/decode capacity modeling and exhaustive search, in bytes/seconds."""
from bisect import bisect_right
from dataclasses import dataclass
import math
import random
from statistics import NormalDist


def positive(value, name, zero=False):
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f'{name} must be finite and {"nonnegative" if zero else "positive"}')


def count(value, name, zero=True):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(not zero):
        raise ValueError(f'{name} must be an integer >= {int(not zero)}')


@dataclass(frozen=True)
class Profile:
    lengths: tuple[float, ...]
    seconds: tuple[float, ...]
    kv_bytes: tuple[float, ...]

    def __post_init__(self):
        if len(self.lengths) < 2 or len(self.seconds) != len(self.lengths) or len(self.kv_bytes) != len(self.lengths):
            raise ValueError('profile requires at least two equally sized columns')
        for name in ('lengths','seconds','kv_bytes'):
            for value in getattr(self,name): positive(value,name)
        if any(a>=b for a,b in zip(self.lengths,self.lengths[1:])):
            raise ValueError('profile lengths must strictly increase')

    def at(self, length):
        if not math.isfinite(length) or not self.lengths[0] <= length <= self.lengths[-1]:
            raise ValueError(f'length {length} outside measured profile range')
        i=min(bisect_right(self.lengths,length)-1,len(self.lengths)-2)
        f=(length-self.lengths[i])/(self.lengths[i+1]-self.lengths[i])
        return tuple(values[i]+f*(values[i+1]-values[i]) for values in (self.seconds,self.kv_bytes))


@dataclass(frozen=True)
class Workload:
    mu: float=9.90
    sigma: float=1.0
    low: float=128
    high: float=128000

    def __post_init__(self):
        positive(self.sigma,'sigma'); positive(self.low,'low'); positive(self.high,'high')
        if not math.isfinite(self.mu) or self.low>=self.high or self.mass(self.low,self.high)<=1e-14:
            raise ValueError('invalid or numerically degenerate truncated lognormal')

    def mass(self,a,b):
        normal=NormalDist()
        return normal.cdf((math.log(b)-self.mu)/self.sigma)-normal.cdf((math.log(a)-self.mu)/self.sigma)

    def moment(self,a,b):
        normal=NormalDist()
        return math.exp(self.mu+self.sigma**2/2)*(normal.cdf((math.log(b)-self.mu-self.sigma**2)/self.sigma)-normal.cdf((math.log(a)-self.mu-self.sigma**2)/self.sigma))

    @property
    def mean(self):
        return self.moment(self.low,self.high)/self.mass(self.low,self.high)

    def split(self,threshold):
        positive(threshold,'threshold',zero=True)
        if threshold<=self.low: return 1.,self.mean,None
        if threshold>=self.high: return 0.,None,self.mean
        short_mass=self.mass(self.low,threshold)
        long_mass=self.mass(threshold,self.high)
        if long_mass <= 1e-14: return 0.,None,self.mean
        if short_mass <= 1e-14: return 1.,self.mean,None
        p=long_mass/(short_mass+long_mass)
        return p, self.moment(threshold,self.high)/long_mass, self.moment(self.low,threshold)/short_mass

    def sample(self,n,seed=0):
        count(n,'sample count')
        rng=random.Random(seed); normal=NormalDist()
        a=normal.cdf((math.log(self.low)-self.mu)/self.sigma)
        mass=self.mass(self.low,self.high)
        return [math.exp(self.mu+self.sigma*normal.inv_cdf(a+mass*rng.random())) for _ in range(n)]


def kv_rate(size_bytes,seconds):
    positive(size_bytes,'size',zero=True); positive(seconds,'seconds')
    return size_bytes/seconds


def required_egress(gpus,parallelism,sizes,times):
    count(gpus,'gpus'); count(parallelism,'parallelism',zero=False)
    if not sizes or len(sizes)!=len(times): raise ValueError('paired nonempty measurements required')
    for s,t in zip(sizes,times): positive(s,'size',zero=True); positive(t,'time')
    return (gpus/parallelism)*sum(sizes)/sum(times)


@dataclass(frozen=True)
class Capacity:
    remote: float
    local: float
    decode: float
    remote_limit: float
    local_limit: float
    total: float
    egress_bytes_s: float


def capacity(p,mean_long,mean_short,remote_profile,local_profile,n_remote,n_prefill,n_decode,bandwidth_bytes_s,batch_max,decode_step_s,output_tokens):
    if not math.isfinite(p) or not 0<=p<=1: raise ValueError('p must be in [0,1]')
    for v,n in [(n_remote,'remote'),(n_prefill,'prefill'),(n_decode,'decode')]: count(v,n)
    positive(bandwidth_bytes_s,'bandwidth',zero=True)
    count(batch_max,'batch',zero=False); positive(decode_step_s,'decode step'); count(output_tokens,'output tokens',zero=False)
    remote=local=kv=0.
    if p:
        seconds,kv=remote_profile.at(mean_long)
        remote=min(n_remote/seconds,bandwidth_bytes_s/kv)
    if p<1:
        seconds,_=local_profile.at(mean_short)
        local=n_prefill/seconds
    decode=n_decode*batch_max/(decode_step_s*output_tokens)
    remote_limit=remote/p if p else math.inf
    local_limit=local/(1-p) if p<1 else math.inf
    total=min(remote_limit,local_limit,decode)
    return Capacity(remote,local,decode,remote_limit,local_limit,total,total*p*kv)


def grid_search(workload,remote_profile,local_profile,n_remote,n_local,bandwidth_bytes_s,batch_max,decode_step_s,output_tokens,thresholds,prefill_counts=None):
    count(n_local,'local instances',zero=False)
    thresholds=list(thresholds)
    counts=list(range(n_local)) if prefill_counts is None else list(prefill_counts)
    if not thresholds or not counts: raise ValueError('empty search grid')
    for n in counts:
        count(n,'prefill instances')
        if n>=n_local: raise ValueError('at least one decode instance required')
    rows=[]
    for t in thresholds:
        p,long,short=workload.split(t)
        for n in counts:
            c=capacity(p,long,short,remote_profile,local_profile,n_remote,n,n_local-n,bandwidth_bytes_s,batch_max,decode_step_s,output_tokens)
            rows.append(dict(threshold=t,p=p,mean_long=long,mean_short=short,prefill_instances=n,decode_instances=n_local-n,throughput=c.total,remote_capacity=c.remote,local_capacity=c.local,decode_capacity=c.decode,egress_gbps=c.egress_bytes_s*8/1e9))
    return max(rows,key=lambda r:r['throughput']), rows
