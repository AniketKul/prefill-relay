"""Open-loop serving workloads with arrival-relative latency and SLO goodput."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import random
from statistics import fmean
import threading
import time
from .analytics import count, positive


POLICIES = ('all-local', 'all-remote', 'selective')


def arrival_offsets(requests, rate, pattern='uniform', burst_size=4):
    count(requests, 'requests', zero=False); positive(rate, 'arrival rate')
    count(burst_size, 'burst size', zero=False)
    if pattern not in ('uniform', 'burst'): raise ValueError('unknown arrival pattern')
    return [i / rate if pattern == 'uniform' else (i // burst_size) * burst_size / rate
            for i in range(requests)]


def skew_prompts(prompts, requests, hot_fraction=.8, seed=0):
    """Repeat a hot prompt while preserving deterministic cold alternatives."""
    count(requests, 'requests', zero=False); positive(hot_fraction, 'hot fraction', zero=True)
    if hot_fraction > 1 or not prompts: raise ValueError('invalid hot prompt distribution')
    rng = random.Random(seed)
    return [list(prompts[0] if rng.random() < hot_fraction else rng.choice(prompts))
            for _ in range(requests)]


def _percentile(values, q):
    ordered = sorted(values)
    if not ordered: return None
    index = (len(ordered) - 1) * q
    low = math.floor(index); high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _summary(rows, elapsed, decode_slo_tps, ttft_slo_s):
    good = [r for r in rows if r['error'] is None]
    passing = [r for r in good if (r['decode_tps'] is None or r['decode_tps'] >= decode_slo_tps)
               and (ttft_slo_s is None or r['ttft_s'] <= ttft_slo_s)]
    ttfts = [r['ttft_s'] for r in good]
    speed = [r['decode_tps'] for r in good if r['decode_tps'] is not None]
    return dict(requests=len(rows), completed=len(good), errors=len(rows)-len(good),
                rejected=sum(r['error'] == 'queue_full' for r in rows), elapsed_s=elapsed,
                throughput_rps=len(good)/elapsed, mean_ttft_s=fmean(ttfts) if ttfts else None,
                p90_ttft_s=_percentile(ttfts, .9), mean_decode_tps=fmean(speed) if speed else None,
                output_tokens_s=sum(len(r['generated_tokens']) for r in good)/elapsed,
                slo_goodput_rps=len(passing)/elapsed, slo_passed=len(passing),
                decode_slo_tps=decode_slo_tps, ttft_slo_s=ttft_slo_s,
                mean_queue_s=fmean(r['queue_s'] for r in rows),
                max_pending=max(r['pending_at_arrival'] for r in rows),
                transferred_bytes=sum(r.get('transferred_bytes', 0) for r in good))


def run_workload(coordinator, prompts, output_tokens=16, policy='selective', rate=1.,
                 pattern='uniform', burst_size=4, max_workers=4, max_pending=None,
                 warm_rounds=0, bandwidth_changes=(), initial_bandwidth_bytes_s=None,
                 decode_slo_tps=40., ttft_slo_s=None, reset=True):
    """Measure offered arrivals independently of service completion.

    ``coordinator.run`` returns ttft_s, decode_s, generated_tokens and transfer
    bytes. Its TTFT starts on invocation; executor waiting is added here. Decode
    speed is (output tokens - 1) / decode_s, excluding the prefill-produced token.
    Warm rounds run before the measurement clock and after the optional reset.
    """
    prompts = [list(tokens) for tokens in prompts]
    if not prompts or any(not tokens for tokens in prompts): raise ValueError('nonempty prompts required')
    for tokens in prompts:
        for token in tokens: count(token, 'token')
    if policy not in POLICIES: raise ValueError('unknown policy')
    count(output_tokens, 'output tokens', zero=False); count(max_workers, 'workers', zero=False)
    count(warm_rounds, 'warm rounds'); positive(decode_slo_tps, 'decode SLO')
    if ttft_slo_s is not None: positive(ttft_slo_s, 'TTFT SLO')
    max_pending = max_workers * 4 if max_pending is None else max_pending
    count(max_pending, 'pending capacity', zero=False)
    offsets = arrival_offsets(len(prompts), rate, pattern, burst_size)
    changes = tuple(bandwidth_changes)
    if changes and initial_bandwidth_bytes_s is None:
        raise ValueError('bandwidth changes require an initial bandwidth for repeatable runs')
    for offset, bandwidth in changes:
        positive(offset, 'change time', zero=True); positive(bandwidth, 'bandwidth', zero=True)
        if offset > offsets[-1]: raise ValueError('bandwidth change follows the last arrival')
    if initial_bandwidth_bytes_s is not None:
        positive(initial_bandwidth_bytes_s, 'initial bandwidth', zero=True)
    if reset: coordinator.reset_caches()
    if initial_bandwidth_bytes_s is not None: coordinator.set_bandwidth(initial_bandwidth_bytes_s)
    for _ in range(warm_rounds):
        for tokens in prompts: coordinator.run(tokens, output_tokens, policy=policy)

    rows = []; futures = []; lock = threading.Lock(); pending = 0
    start = time.perf_counter()

    def execute(index, offset, arrived, pending_at_arrival):
        nonlocal pending
        entered = time.perf_counter()
        queue = max(0., entered-start-offset)
        row = dict(index=index, prompt_tokens=len(prompts[index]), scheduled_s=offset,
                   arrived_s=arrived, started_s=entered-start, queue_s=queue,
                   dispatch_delay_s=max(0., arrived-offset),
                   executor_queue_s=max(0., entered-start-arrived),
                   pending_at_arrival=pending_at_arrival)
        try:
            result = coordinator.run(prompts[index], output_tokens, policy=policy)
            ttft = result['ttft_s']; decode = result['decode_s']; generated = result['generated_tokens']
            positive(ttft, 'measured TTFT', zero=True); positive(decode, 'measured decode time', zero=True)
            if not isinstance(generated, list) or not generated or len(generated) > output_tokens:
                raise ValueError('invalid generated token count')
            for token in generated: count(token, 'generated token')
            if len(generated) > 1 and decode == 0: raise ValueError('missing decode timing')
            count(result['transferred_bytes'], 'transfer bytes')
            row.update(result)
            row.update(error=None, ttft_s=ttft+queue, queue_s=queue,
                       decode_tps=(len(generated)-1)/decode if len(generated)>1 else None)
        except Exception as error:
            row.update(error=f'{type(error).__name__}: {error}')
        finally:
            row['finished_s'] = time.perf_counter()-start
            row['total_s'] = row['finished_s']-offset
            with lock: pending -= 1
        return row

    events = [(offset, 1, index) for index, offset in enumerate(offsets)]
    events.extend((offset, 0, bandwidth) for offset, bandwidth in changes)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for offset, kind, value in sorted(events):
            delay = start+offset-time.perf_counter()
            if delay > 0: time.sleep(delay)
            if kind == 0:
                coordinator.set_bandwidth(value)
                continue
            with lock:
                if pending >= max_pending:
                    now = time.perf_counter()-start
                    rows.append(dict(index=value, prompt_tokens=len(prompts[value]),
                                     scheduled_s=offset, arrived_s=now, started_s=now,
                                     finished_s=now, total_s=now-offset, queue_s=max(0., now-offset),
                                     dispatch_delay_s=max(0., now-offset), executor_queue_s=0.,
                                     pending_at_arrival=pending, error='queue_full'))
                    continue
                pending += 1
                pending_at_arrival = pending
            futures.append(executor.submit(execute, value, offset,
                                           time.perf_counter()-start, pending_at_arrival))
        rows.extend(future.result() for future in futures)
    rows.sort(key=lambda row: row['index'])
    elapsed = max(max(row['finished_s'] for row in rows), 1e-9)
    return dict(kind='measured serving workload', policy=policy, pattern=pattern, rate=rate,
                inputs_sha256=hashlib.sha256(json.dumps(prompts,separators=(',',':')).encode()).hexdigest(),
                warm_rounds=warm_rounds, max_workers=max_workers, max_pending=max_pending,
                initial_bandwidth_bytes_s=initial_bandwidth_bytes_s,
                bandwidth_changes=[list(change) for change in changes],
                summary=_summary(rows, elapsed, decode_slo_tps, ttft_slo_s), requests=rows)


def compare_policies(coordinator, prompts, policies=POLICIES, **kwargs):
    """Reset caches before each policy; preserve prompts, arrivals and warmup."""
    if 'policy' in kwargs or 'reset' in kwargs: raise ValueError('comparison owns policy and reset')
    policies = tuple(policies)
    if not policies or len(policies) != len(set(policies)): raise ValueError('distinct policies required')
    prompts = [list(tokens) for tokens in prompts]
    return {policy: run_workload(coordinator, prompts, policy=policy, reset=True, **kwargs)
            for policy in policies}
