"""Measured pretrained service-time and cache-size profiles for capacity search."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from .hf_backend import HFHybrid
from .hf_state import state_bytes


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def profile_backend(backend, lengths=(1024, 8192, 32768, 131072), batches=(1,),
                    repeats=3, chunk_size=4096) -> dict:
    """Measure warm prefill and batched decode; record every configuration.

    The dense cache estimate changes only the attention allocation, not trained
    weights. It is a byte accounting comparison, never a speed or quality claim.
    """
    if any(type(n) is not int or n <= 0 for n in (*lengths, *batches, repeats, chunk_size)):
        raise ValueError('lengths, batches, repeats and chunk size must be positive integers')
    if len(set(lengths)) != len(lengths) or sorted(lengths) != list(lengths):
        raise ValueError('lengths must be strictly increasing')
    if max(lengths) + 1 > backend.max_context:
        raise ValueError('profile and decode exceed model context limit')
    seed = backend.tokenizer('A reusable prefix reduces repeated computation. ', return_tensors='pt').input_ids[0]
    rows = []
    for batch in batches:
        for length in lengths:
            vector = seed.repeat((length + seed.numel() - 1) // seed.numel())[:length].to(backend.device)
            tokens = vector if batch == 1 else vector.unsqueeze(0).repeat(batch, 1)
            # Each length has an untimed warmup to exclude shape-specific compilation.
            backend.prefill(tokens, chunk_size=chunk_size)
            synchronize(backend.device)
            if backend.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(backend.device)
            timings, decode_times = [], []
            for _ in range(repeats):
                start = time.perf_counter()
                logits, state = backend.prefill(tokens, chunk_size=chunk_size)
                synchronize(backend.device)
                timings.append(time.perf_counter() - start)
                decode_steps = min(8, backend.max_context-length)
                start = time.perf_counter()
                generated = backend.generate(state, decode_steps+1)
                synchronize(backend.device)
                decode_times.append((time.perf_counter() - start)/decode_steps)
                if not torch.isfinite(logits).all():
                    raise RuntimeError('nonfinite inference logits')
            kv_bytes = state_bytes(state)
            cfg = backend.config
            element_size = next(backend.model.parameters()).element_size()
            dense_estimate = batch * length * cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * element_size
            full_layers = sum(kind == 'full_attention' for kind in cfg.layer_types)
            attention_bytes = batch * length * full_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * element_size
            row = {'length': length, 'batch': batch, 'seconds': statistics.median(timings),
                   'samples_seconds': timings, 'decode_step_seconds': statistics.median(decode_times),
                   'decode_samples_seconds': decode_times, 'decode_steps_per_sample': decode_steps, 'kv_bytes': kv_bytes,
                   'full_attention_bytes': attention_bytes, 'linear_state_bytes': kv_bytes - attention_bytes,
                   'dense_cache_estimate_bytes': dense_estimate,
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated(backend.device) if backend.device.type == 'cuda' else None}
            rows.append(row)
            print(json.dumps({'measurement': row}), flush=True)
            del state, logits, generated
    single = [row for row in rows if row['batch'] == 1]
    return {'kind': 'measured pretrained inference on one device', 'model': backend.model_id,
            'revision': backend.revision, 'dtype': str(backend.model.dtype), 'torch': torch.__version__,
            'python': platform.python_version(),
            'dependencies': {name: importlib.metadata.version(name) for name in ('transformers', 'flash-linear-attention', 'fla-core')},
            'device': torch.cuda.get_device_name(backend.device) if backend.device.type == 'cuda' else str(backend.device),
            'chunk_size': chunk_size, 'kernels': backend.kernel_provenance, 'rows': rows,
            'profile': {'lengths': [r['length'] for r in single], 'seconds': [r['seconds'] for r in single],
                        'kv_bytes': [r['kv_bytes'] for r in single]},
            'decode_method': 'one native-cache restoration per continuation; average of up to eight sequential steps',
            'dense_comparison': 'configuration byte estimate only; no dense-model timing or quality measurement'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16', choices=['bfloat16', 'float16', 'float32'])
    parser.add_argument('--lengths', type=int, nargs='+', default=[1024, 8192, 32768, 131072])
    parser.add_argument('--batches', type=int, nargs='+', default=[1])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    backend = HFHybrid(args.model, args.revision, args.device, args.dtype)
    result = {}
    if args.validate or args.validate_only:
        from .hf_validate import validate_backend
        result['validation'] = validate_backend(backend)
        print(json.dumps({'validation': result['validation']}), flush=True)
    if not args.validate_only:
        result['measurements'] = profile_backend(backend, args.lengths, args.batches, args.repeats, args.chunk_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
