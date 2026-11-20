"""Measure raw attention-state production rates on two pinned trained checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
import statistics
import time
from pathlib import Path

import torch


MODELS = (
    ('hybrid', 'Qwen/Qwen3.5-0.8B', '2fc06364715b967f1860aea9cf38778875588b17'),
    ('dense', 'ibm-granite/granite-3.3-2b-instruct', '707f574c62054322f6b5b04b6d075f0a8f05e0f0'),
)


def cache_sizes(cache, layer_types, length):
    """Count transferable tensor bytes, excluding tokens, logits and metadata."""
    full = recurrent = 0
    if len(cache.layers) != len(layer_types):
        raise ValueError('incomplete cache')
    for kind, layer in zip(layer_types, cache.layers):
        if kind == 'full_attention':
            key, value = layer.keys, layer.values
            if key.ndim != 4 or key.shape != value.shape or key.shape[-2] != length:
                raise ValueError('full attention cache does not cover prompt length')
            full += key.nbytes + value.nbytes
        elif kind == 'linear_attention':
            conv, state = layer.conv_states[0], layer.recurrent_states[0]
            if not conv.numel() or not state.numel():
                raise ValueError('empty recurrent state')
            recurrent += conv.nbytes + state.nbytes
        else:
            raise ValueError(f'unsupported cache layer type {kind}')
    return full, recurrent


@torch.inference_mode()
def measure(model, tokens, chunk_size, repeats):
    from transformers import DynamicCache
    cfg = model.config
    layer_types = getattr(cfg, 'layer_types', None) or ['full_attention'] * cfg.num_hidden_layers
    samples = []
    for run in range(repeats + 1):
        cache = DynamicCache(config=cfg)
        torch.cuda.synchronize(tokens.device)
        torch.cuda.reset_peak_memory_stats(tokens.device)
        start = time.perf_counter()
        for chunk in tokens.split(chunk_size, dim=-1):
            output = model(input_ids=chunk, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cache = output.past_key_values
        torch.cuda.synchronize(tokens.device)
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated(tokens.device)
        if not torch.isfinite(output.logits).all():
            raise ValueError('nonfinite final logits')
        full, recurrent = cache_sizes(cache, layer_types, tokens.shape[-1])
        if run:
            samples.append({'seconds': elapsed, 'peak_gpu_bytes': peak})
        del cache, output
    median = statistics.median(row['seconds'] for row in samples)
    return {'length': tokens.shape[-1], 'full_kv_bytes': full, 'recurrent_bytes': recurrent,
            'state_bytes': full + recurrent, 'samples': samples, 'median_seconds': median,
            'kv_gbps': 8 * (full + recurrent) / median / 1e9}


def main():
    from transformers import GraniteForCausalLM, Qwen3_5ForCausalLM
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lengths', type=int, nargs='+', default=[1024, 8192, 32768, 131072])
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.lengths, args.chunk_size, args.repeats) < 1 or sorted(set(args.lengths)) != args.lengths:
        parser.error('lengths must be positive and strictly increasing, and chunk size and repeats positive')
    result = {'device': torch.cuda.get_device_name(), 'torch': str(torch.__version__),
              'dependencies': {p: importlib.metadata.version(p) for p in ('transformers', 'flash-linear-attention')},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'dtype': 'bfloat16', 'attention': 'sdpa', 'batch': 1,
              'input_ids': 'arange(length) % 1024', 'repeats': args.repeats,
              'chunk_size': args.chunk_size, 'lengths': args.lengths, 'models': []}
    try:
        result['dependencies']['causal-conv1d'] = importlib.metadata.version('causal-conv1d')
    except importlib.metadata.PackageNotFoundError:
        pass
    for (kind, name, revision), loader in zip(MODELS, (Qwen3_5ForCausalLM, GraniteForCausalLM)):
        model = loader.from_pretrained(name, revision=revision, dtype=torch.bfloat16,
                                       attn_implementation='sdpa', trust_remote_code=False).to('cuda').eval()
        cfg = model.config
        if max(args.lengths) > cfg.max_position_embeddings:
            parser.error(f'requested length exceeds context limit of {name}')
        data = {'architecture': kind, 'model': name, 'revision': revision,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'max_context': cfg.max_position_embeddings, 'layers': cfg.num_hidden_layers,
                'num_key_value_heads': cfg.num_key_value_heads, 'rows': []}
        for length in args.lengths:
            tokens = (torch.arange(length, device='cuda') % 1024).unsqueeze(0)
            row = measure(model, tokens, args.chunk_size, args.repeats)
            data['rows'].append(row)
            print(json.dumps({'model': name, **row}), flush=True)
        result['models'].append(data)
        del model
        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
