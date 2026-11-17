"""Small, untrained 3:1 hybrid used to verify real tensor cache handoff.

This untrained linear-attention/full-attention network validates cache
correctness; it is not a production language model.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import re
import struct
from collections.abc import Callable

import torch
from torch import Tensor, nn
import torch.nn.functional as F

State = dict[str, Tensor]
_DTYPES = {'float32': torch.float32, 'int64': torch.int64, 'uint8': torch.uint8}
_SIZES = {'float32': 4, 'int64': 8, 'uint8': 1}
_MAGIC = b'PRFKV001'
_MAX_BYTES = 256 * 1024 * 1024


def encode_state(state: State, max_bytes: int = _MAX_BYTES) -> bytes:
    """Serialize tensors as bounded JSON metadata + raw bytes + SHA256, no pickle."""
    if not state or len(state) > 128:
        raise ValueError('expected 1..128 tensors')
    metadata, chunks, total = [], [], 0
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.]{1,80}', name):
            raise ValueError('invalid tensor name')
        if not isinstance(value, Tensor) or value.layout != torch.strided:
            raise ValueError('expected dense tensor')
        dtype = str(value.dtype).removeprefix('torch.')
        if dtype not in _DTYPES or value.ndim > 4:
            raise ValueError('unsupported tensor dtype or rank')
        size = value.numel() * value.element_size()
        total += size
        if total > max_bytes:
            raise ValueError('state exceeds byte limit')
        value = value.detach().cpu().contiguous()
        chunks.append(ctypes.string_at(value.data_ptr(), size) if size else b'')
        metadata.append({'name': name, 'dtype': dtype, 'shape': list(value.shape), 'bytes': size})
    header = json.dumps(metadata, separators=(',', ':')).encode('ascii')
    body = _MAGIC + struct.pack('!I', len(header)) + header + b''.join(chunks)
    if len(header) > 65536 or len(body) + 32 > max_bytes:
        raise ValueError('state exceeds byte limit')
    return body + hashlib.sha256(body).digest()


def decode_state(payload: bytes, device: str | torch.device = 'cpu',
                 max_bytes: int = _MAX_BYTES) -> State:
    """Validate all dimensions and byte counts before allocating tensors."""
    if len(payload) > max_bytes or len(payload) < 44 or payload[:8] != _MAGIC:
        raise ValueError('invalid state envelope or byte limit')
    body, checksum = payload[:-32], payload[-32:]
    if hashlib.sha256(body).digest() != checksum:
        raise ValueError('state checksum mismatch')
    header_size = struct.unpack('!I', body[8:12])[0]
    if header_size > 65536 or 12 + header_size > len(body):
        raise ValueError('invalid metadata length')
    try:
        metadata = json.loads(body[12:12 + header_size])
    except (ValueError, UnicodeError) as error:
        raise ValueError('invalid state metadata') from error
    if not isinstance(metadata, list) or not 1 <= len(metadata) <= 128:
        raise ValueError('invalid tensor list')
    offset, entries, seen = 12 + header_size, [], set()
    for entry in metadata:
        if not isinstance(entry, dict) or set(entry) != {'name', 'dtype', 'shape', 'bytes'}:
            raise ValueError('invalid tensor metadata')
        name, dtype, shape, size = (entry[k] for k in ('name', 'dtype', 'shape', 'bytes'))
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.]{1,80}', name) or name in seen:
            raise ValueError('invalid or duplicate tensor name')
        if not isinstance(dtype, str) or dtype not in _DTYPES:
            raise ValueError('invalid tensor dtype')
        if not isinstance(shape, list) or len(shape) > 4 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('invalid tensor dimensions')
        expected = math.prod(shape) * _SIZES[dtype]
        if type(size) is not int or size != expected or offset + size > len(body):
            raise ValueError('invalid tensor byte count')
        entries.append((name, dtype, shape, offset, size))
        seen.add(name)
        offset += size
    if offset != len(body):
        raise ValueError('unexpected trailing state bytes')
    state = {}
    for name, dtype, shape, offset, size in entries:
        if size:
            value = torch.frombuffer(bytearray(body[offset:offset + size]), dtype=_DTYPES[dtype]).reshape(shape)
        else:
            value = torch.empty(shape, dtype=_DTYPES[dtype])
        state[name] = value.to(device)
    return state


def _length(state: State) -> int:
    length = state.get('length')
    if not isinstance(length, Tensor) or length.dtype != torch.int64 or length.numel() != 1:
        raise ValueError('cache requires int64 length')
    result = int(length.item())
    if result < 0:
        raise ValueError('cache length must be nonnegative')
    return result


def _digest(tokens: Tensor) -> Tensor:
    raw = encode_state({'tokens': tokens})
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8, device=tokens.device)


def delta(state: State, prefix_length: int) -> State:
    """Send missing full-attention KV and final recurrent checkpoints.

    prefix_length is what the DESTINATION already holds; it need not equal
    the prefix reused by the producer. The final recurrent state is indivisible.
    """
    length = _length(state)
    if type(prefix_length) is not int or not 0 <= prefix_length <= length:
        raise ValueError('invalid destination prefix length')
    if state.get('tokens') is None or state['tokens'].shape != (length,):
        raise ValueError('invalid cache token identity')
    change = dict(state)
    change['base_length'] = torch.tensor(prefix_length, dtype=torch.int64, device=state['length'].device)
    change['base_digest'] = _digest(state['tokens'][:prefix_length])
    change['tokens'] = state['tokens'][prefix_length:]
    for name in ('layer.3.key', 'layer.3.value'):
        if name in state:
            if state[name].ndim != 2 or state[name].shape[0] != length:
                raise ValueError('invalid full-attention cache length')
            change[name] = state[name][prefix_length:]
    return change


def merge(base: State | None, change: State) -> State:
    """Reconstruct a transferred cache, rejecting a different or missing prefix."""
    if 'base_length' not in change or 'base_digest' not in change:
        raise ValueError('missing transfer prefix metadata')
    prefix = _length({'length': change['base_length']})
    total = _length(change)
    if prefix > total or (base is None and prefix) or (base is not None and _length(base) != prefix):
        raise ValueError('destination prefix length mismatch')
    suffix = change.get('tokens')
    if suffix is None or suffix.dtype != torch.int64 or suffix.shape != (total - prefix,):
        raise ValueError('invalid transferred token suffix')
    prefix_tokens = suffix[:0] if base is None else base['tokens']
    if not torch.equal(_digest(prefix_tokens).cpu(), change['base_digest'].cpu()):
        raise ValueError('destination prefix identity mismatch')
    if base is not None and not torch.equal(base['model_config'].cpu(), change['model_config'].cpu()):
        raise ValueError('different model cache')
    state = {k: v for k, v in change.items() if k not in ('base_length', 'base_digest')}
    state['tokens'] = torch.cat((prefix_tokens.to(suffix.device), suffix))
    for name in ('layer.3.key', 'layer.3.value'):
        if name in change:
            value = change[name]
            if value.ndim != 2 or value.shape[0] != total - prefix:
                raise ValueError('invalid full-attention suffix shape')
            previous = value[:0] if base is None else base[name].to(value.device)
            if previous.shape != (prefix, value.shape[1]):
                raise ValueError('invalid destination full-attention cache')
            state[name] = torch.cat((previous, value))
    return state


class TinyHybrid(nn.Module):
    """Deterministic, untrained 3 linear : 1 full causal attention model."""

    def __init__(self, vocab_size: int = 256, width: int = 32, seed: int = 0):
        super().__init__()
        if type(vocab_size) is not int or type(width) is not int or not 2 <= vocab_size <= 65536 or not 1 <= width <= 256:
            raise ValueError('invalid vocabulary size or width')
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('invalid seed')
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(vocab_size, width)
            self.qkv = nn.ModuleList(nn.Linear(width, 3 * width, bias=False) for _ in range(4))
            self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(4))
            self.projections = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(4))
            self.head = nn.Linear(width, vocab_size, bias=False)
        self.register_buffer('config', torch.tensor([vocab_size, width, seed], dtype=torch.int64))
        self.width = width
        self.vocab_size = vocab_size

    def _validate_cache(self, cache: State, device: torch.device) -> int:
        length = _length(cache)
        expected = {'length', 'tokens', 'model_config', 'layer.3.key', 'layer.3.value'}
        expected.update(f'layer.{i}.{name}' for i in range(3) for name in ('matrix', 'normalizer'))
        if set(cache) != expected or not torch.equal(cache['model_config'], self.config):
            raise ValueError('incomplete cache or different model configuration')
        if cache['tokens'].shape != (length,) or cache['tokens'].dtype != torch.int64:
            raise ValueError('invalid cache token identity')
        for name, value in cache.items():
            if value.device != device:
                raise ValueError('cache and model must be on the same device')
            if name.endswith('.matrix'):
                shape = (self.width, self.width)
            elif name.endswith('.normalizer'):
                shape = (self.width,)
            elif name.endswith(('.key', '.value')):
                shape = (length, self.width)
            else:
                continue
            if value.shape != shape or value.dtype != self.embedding.weight.dtype or not torch.isfinite(value).all():
                raise ValueError('invalid cache tensor shape, dtype or values')
        return length

    @torch.inference_mode()
    def forward(self, tokens: Tensor, cache: State | None = None,
                on_layer: Callable[[int, State], None] | None = None) -> tuple[Tensor, State]:
        if not isinstance(tokens, Tensor) or tokens.ndim != 1 or tokens.dtype != torch.int64 or not tokens.numel():
            raise ValueError('tokens must be a nonempty int64 vector')
        if tokens.device != self.embedding.weight.device or (tokens < 0).any() or (tokens >= self.vocab_size).any():
            raise ValueError('invalid token id or device')
        start = 0 if cache is None else self._validate_cache(cache, tokens.device)
        length = start + tokens.numel()
        # shortcut: bounded dense attention oracle, replace with production kernels for long contexts.
        if length > 8192:
            raise ValueError('tiny correctness backend supports at most 8192 tokens')
        all_tokens = tokens if cache is None else torch.cat((cache['tokens'], tokens))
        metadata = {'length': torch.tensor(length, dtype=torch.int64, device=tokens.device),
                    'tokens': all_tokens, 'model_config': self.config.clone()}
        state = dict(metadata)
        hidden = self.embedding(tokens)
        for i in range(4):
            q, k, v = self.qkv[i](self.norms[i](hidden)).chunk(3, dim=-1)
            if i < 3:
                q, k = F.elu(q) + 1, F.elu(k) + 1
                matrices = (k.unsqueeze(-1) * v.unsqueeze(-2)).cumsum(dim=0)
                normalizers = k.cumsum(dim=0)
                if cache is not None:
                    matrices = matrices + cache[f'layer.{i}.matrix']
                    normalizers = normalizers + cache[f'layer.{i}.normalizer']
                output = torch.einsum('td,tdv->tv', q, matrices)
                output = output / (q * normalizers).sum(dim=-1, keepdim=True).clamp_min(1e-6)
                layer = {f'layer.{i}.matrix': matrices[-1].clone(),
                         f'layer.{i}.normalizer': normalizers[-1].clone()}
            else:
                if cache is not None:
                    k = torch.cat((cache['layer.3.key'], k))
                    v = torch.cat((cache['layer.3.value'], v))
                scores = q @ k.T / math.sqrt(self.width)
                mask = torch.arange(length, device=tokens.device)[None, :] > torch.arange(start, length, device=tokens.device)[:, None]
                output = torch.softmax(scores.masked_fill(mask, float('-inf')), dim=-1) @ v
                layer = {'layer.3.key': k, 'layer.3.value': v}
            hidden = hidden + self.projections[i](output)
            state.update(layer)
            if on_layer is not None:
                on_layer(i, {**metadata, **layer})
        return self.head(hidden), state
