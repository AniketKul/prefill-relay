"""Measure decode batch capacity on the same device and model as a prefill profile."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from .analytics import count, positive


def select_decode_batch(rows, required_lengths, min_tokens_s, max_batch=4):
    """Largest tested size meeting the per-sequence SLO at every required length."""
    positive(min_tokens_s, 'minimum tokens per second')
    count(max_batch, 'maximum batch', zero=False)
    lengths = sorted(set(required_lengths))
    if not lengths:
        raise ValueError('at least one decode context length required')
    for length in lengths:
        count(length, 'decode context length', zero=False)
    times = {}
    for row in rows:
        batch, length = row.get('batch', 1), row['length']
        count(batch, 'profile batch', zero=False)
        if batch > max_batch or length not in lengths:
            continue
        if (batch, length) in times:
            raise ValueError('duplicate decode timing for batch and context length')
        step = row['decode_step_seconds']
        positive(step, 'measured decode step')
        times[batch, length] = step
    by_batch = []
    best, best_step = 0, None
    for batch in range(1, max_batch + 1):
        missing = [n for n in lengths if (batch, n) not in times]
        worst = None if missing else max(times[batch, n] for n in lengths)
        meets = worst is not None and worst <= 1 / min_tokens_s
        by_batch.append(dict(batch=batch, missing_lengths=missing, worst_step_s=worst,
                             min_tokens_s=1 / worst if worst else None, eligible=meets))
        if meets:
            best, best_step = batch, worst
    return dict(batch_max=best, decode_step_s=best_step, by_batch=by_batch,
                required_lengths=lengths, min_tokens_s=min_tokens_s)


def measure(profile_file, lengths, max_batch, steps, repeats, decode_graphs=False, reserve_decode_kv=False):
    import torch
    from .hf_backend import HFHybrid

    if reserve_decode_kv and not decode_graphs:
        raise ValueError('decode KV reservation requires decode graphs')
    raw = profile_file.read_bytes()
    base = json.loads(raw)['measurements']
    lengths = sorted(set(lengths))
    for value in [max_batch, steps, repeats, *lengths]:
        count(value, 'measurement arguments', zero=False)
    if not lengths:
        raise ValueError('at least one context length required')
    backend = HFHybrid(base['model'], base['revision'], device='cuda',
                       dtype=base['dtype'].removeprefix('torch.'))
    device = backend.device
    if (base['device'] != torch.cuda.get_device_name(device) or base['torch'] != torch.__version__
            or base['revision'] != backend.revision or base.get('kernels') != backend.kernel_provenance):
        raise ValueError('prefill profile and current model, device or kernels differ')
    if max(lengths) + steps > backend.max_context:
        raise ValueError('requested measurement exceeds model context window')
    if decode_graphs:
        backend.enable_decode_graphs(reserve_kv=reserve_decode_kv)
    seed = backend.tokenizer('A reusable prefix reduces repeated computation. ', return_tensors='pt').input_ids[0]
    rows = []
    for length in lengths:
        tokens = seed.repeat((length + seed.numel() - 1) // seed.numel())[:length].to(device)
        _, single = backend.prefill(tokens, chunk_size=base['chunk_size'])
        for batch in range(1, max_batch + 1):
            # Materialize the same shapes as the serving batcher before timing decode.
            state = single
            if batch > 1:
                state = dict(single)
                for key, value in single.items():
                    if key in ('tokens', 'last_logits'):
                        state[key] = torch.stack([value] * batch)
                    elif key.startswith('layer.'):
                        state[key] = torch.cat([value] * batch, dim=0)
            backend.generate(state, steps + 1)
            torch.cuda.synchronize(device)
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                generated = backend.generate(state, steps + 1)
                torch.cuda.synchronize(device)
                samples.append((time.perf_counter() - start) / steps)
                if generated.shape != (batch, steps + 1):
                    raise ValueError('unexpected generated token shape')
            row = dict(batch=batch, length=length, decode_step_seconds=statistics.median(samples),
                       decode_samples_seconds=samples, decode_steps_per_sample=steps)
            rows.append(row)
            print(json.dumps(row), flush=True)
            del state
        del single
    files = ['calibration.py', 'hf_backend.py', 'hf_decode.py', 'hf_state.py']
    if decode_graphs:
        files.append('mlp_graphs.py')
    if reserve_decode_kv:
        files.append('hf_decode_layer.py')
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in files}
    return {'measurements': {**base, 'rows': rows, 'kind': 'prefill profile plus native decode batch calibration',
            'decode_method': 'warm synchronized native continuations; median seconds per generated step',
            'temperature': 'greedy', 'repeats': repeats, 'prefill_profile_sha256': hashlib.sha256(raw).hexdigest(),
            'decode_context_lengths': lengths, 'decode_graphs': decode_graphs, 'reserve_decode_kv': reserve_decode_kv, 'source_sha256': sources}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    parser.add_argument('--lengths', nargs='+', type=int, default=[128, 1024, 8192, 32768, 131072])
    parser.add_argument('--max-batch', type=int, default=4)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--decode-graphs', action='store_true')
    parser.add_argument('--reserve-decode-kv', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.profile, args.lengths, args.max_batch, args.steps, args.repeats, args.decode_graphs, args.reserve_decode_kv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
