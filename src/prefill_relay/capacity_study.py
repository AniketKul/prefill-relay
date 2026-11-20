"""Compare placement and allocation using device profiles and a long-context workload."""
import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path
from statistics import fmean

from .analytics import Profile, Workload, count, grid_search, positive, required_egress
from .calibration import select_decode_batch


def modeled_ttft(lengths, arrival_rate, row, profile, remote_instances, link_bytes_s, batch, decode_s):
    """Uniform arrivals and FIFO stage queues. Transfer starts at prefill completion."""
    producers = {False: [0.] * row['prefill_instances'], True: [0.] * remote_instances}
    ready = []
    for idx, length in enumerate(lengths):
        remote = length > row['threshold']
        # A producer is available for the next request as soon as computation ends.
        pool = producers[remote]
        start = max(idx / arrival_rate, pool[0])
        prefill, size = profile.at(length)
        end = start + prefill
        heapq.heapreplace(pool, end)
        ready.append((end, idx, size if remote else 0))
    link_free = 0.
    arrivals = []
    for end, idx, size in sorted(ready):
        if size:
            end = max(end, link_free) + size / link_bytes_s
            link_free = end
        arrivals.append((end, idx))
    # Ideal continuous batching: each decode slot remains occupied for one response.
    slots = [0.] * (row['decode_instances'] * batch)
    ttft = []
    for end, idx in sorted(arrivals):
        first_token = max(end, slots[0])
        ttft.append(first_token - idx / arrival_rate)
        heapq.heapreplace(slots, first_token + decode_s)
    return {'mean_s': fmean(ttft), 'p90_s': sorted(ttft)[math.ceil(.9 * len(ttft)) - 1]}


def study(measurements, contexts, slo, settings):
    profile = Profile(**measurements['profile'])
    workload = Workload(mu=settings['mu'], sigma=settings['sigma'], low=settings['low'], high=settings['high'])
    profile.at(workload.low)
    profile.at(workload.high)
    selection = select_decode_batch(measurements['rows'], contexts, slo, settings['max_batch'])
    result = {'calibration': selection, 'mean_input_tokens': workload.mean, 'policies': {}}
    if selection['batch_max'] == 0:
        result['status'] = 'no measured batch meets decode SLO at every required length'
        return result
    result['status'] = 'modeled'
    remote, local = settings['remote_instances'], settings['local_instances']
    for name in ('remote_instances', 'local_instances', 'output_tokens', 'samples', 'threshold_step', 'low', 'high'):
        count(settings[name], name, zero=False)
    for name in ('remote_unit_cost', 'local_unit_cost', 'link_gbps', 'offered_fraction'):
        positive(settings[name], name)
    if settings['offered_fraction'] >= 1:
        raise ValueError('offered fraction must be below one')
    budget = remote * settings['remote_unit_cost'] + local * settings['local_unit_cost']
    equal_budget_local = math.floor(budget / settings['local_unit_cost'] + 1e-12)
    if min(remote + local, equal_budget_local) < 2:
        raise ValueError('local-only baseline needs separate prefill and decode capacity')
    threshold_grid = [0, *range(settings['threshold_step'], workload.high, settings['threshold_step']), workload.high]
    batch, step = selection['batch_max'], selection['decode_step_s']
    link = settings['link_gbps'] * 1e9 / 8
    policies = {'selective': (remote, local, threshold_grid, None),
                'all_remote': (remote, local, [0], [0]),
                'all_local_same_count': (0, remote + local, [workload.high], None),
                'all_local_same_budget': (0, equal_budget_local, [workload.high], None)}
    for policy, (remote_count, local_count, thresholds, counts) in policies.items():
        best, grid = grid_search(workload, profile, profile, remote_count, local_count, link,
                                 batch, step, settings['output_tokens'], thresholds, counts)
        best = dict(best)
        best['remote_instances'] = remote_count
        best['resource_cost'] = remote_count * settings['remote_unit_cost'] + local_count * settings['local_unit_cost']
        best['budget_units'] = budget
        best['egress_fraction'] = best['egress_gbps'] / settings['link_gbps']
        result['policies'][policy] = best
        if policy == 'selective':
            # Store both one-dimensional sections of the exhaustive two-dimensional grid.
            result['configurations_searched'] = len(grid)
            result['threshold_sweep'] = [[r['threshold'], r['throughput']] for r in grid
                                         if r['prefill_instances'] == best['prefill_instances']]
            result['allocation_sweep'] = [[r['prefill_instances'], r['throughput']] for r in grid
                                          if r['threshold'] == best['threshold']]
    lengths = [round(length) for length in workload.sample(settings['samples'], settings['seed'])]
    result['input_sha256'] = hashlib.sha256(json.dumps(lengths, separators=(',', ':')).encode()).hexdigest()
    offered = min(row['throughput'] for row in result['policies'].values()) * settings['offered_fraction']
    result['offered_requests_s'] = offered
    count_baseline = result['policies']['all_local_same_count']['throughput']
    budget_baseline = result['policies']['all_local_same_budget']['throughput']
    for row in result['policies'].values():
        row['throughput_ratio_same_count'] = row['throughput'] / count_baseline
        row['throughput_ratio_same_budget'] = row['throughput'] / budget_baseline
        offloaded = [profile.at(n) for n in lengths if n > row['threshold']]
        if offloaded:
            seconds, sizes = zip(*offloaded)
            row['compute_saturated_egress_gbps'] = required_egress(row['remote_instances'], 1, sizes, seconds) * 8 / 1e9
        else:
            row['compute_saturated_egress_gbps'] = 0.
        row['modeled_ttft'] = modeled_ttft(lengths, offered, row, profile, row['remote_instances'], link,
                                          batch, step * settings['output_tokens'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--slo', nargs='+', type=float, default=[40., 25.])
    parser.add_argument('--contexts', nargs='+', type=int)
    parser.add_argument('--max-batch', type=int, default=4)
    parser.add_argument('--mu', type=float, default=9.90)
    parser.add_argument('--sigma', type=float, default=1.)
    parser.add_argument('--low', type=int, default=128)
    parser.add_argument('--high', type=int, default=128000)
    parser.add_argument('--threshold-step', type=int, default=256)
    parser.add_argument('--remote-instances', type=int, default=4)
    parser.add_argument('--local-instances', type=int, default=8)
    parser.add_argument('--output-tokens', type=int, default=1024)
    parser.add_argument('--link-gbps', type=float, default=100.)
    parser.add_argument('--remote-unit-cost', type=float, default=1.)
    parser.add_argument('--local-unit-cost', type=float, default=1.)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--offered-fraction', type=float, default=.8)
    args = parser.parse_args()
    raw = args.calibration.read_bytes()
    measurement = json.loads(raw)['measurements']
    contexts = args.contexts or measurement['decode_context_lengths']
    settings = {k: v for k, v in vars(args).items() if k not in ('calibration', 'output', 'slo', 'contexts')}
    report = {'kind': 'capacity and FIFO latency estimates from a single-device profile',
              'model': measurement['model'], 'revision': measurement['revision'], 'device': measurement['device'],
              'calibration_sha256': hashlib.sha256(raw).hexdigest(), 'settings': settings,
              'source_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('analytics.py', 'capacity_study.py', 'calibration.py')},
              'targets': [study(measurement, contexts, target, settings) for target in args.slo]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
