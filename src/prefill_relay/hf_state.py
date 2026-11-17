"""Bounded, pickle-free tensors and exact-prefix transfers for trained models."""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import re
import struct

import torch
from torch import Tensor

State = dict[str, Tensor]
_MAGIC = b'PRFHF001'
_MAX_BYTES = 4 * 1024**3
_DTYPES = {name: getattr(torch, name) for name in ('float32', 'float16', 'bfloat16', 'int64', 'uint8')}
_SIZES = {'float32': 4, 'float16': 2, 'bfloat16': 2, 'int64': 8, 'uint8': 1}


def encode(state: State, max_bytes: int = _MAX_BYTES) -> bytes:
    """Encode dense tensors with a SHA256 integrity checksum; never pickle."""
    if not state or len(state) > 1024:
        raise ValueError('expected 1..1024 tensors')
    metadata, chunks, total = [], [], 0
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.]{1,80}', name):
            raise ValueError('invalid tensor name')
        if not isinstance(value, Tensor) or value.layout != torch.strided:
            raise ValueError('expected dense tensor')
        dtype = str(value.dtype).removeprefix('torch.')
        if dtype not in _DTYPES or value.ndim > 5:
            raise ValueError('unsupported tensor dtype or rank')
        size = value.numel() * value.element_size()
        total += size
        if total > max_bytes:
            raise ValueError('state exceeds byte limit')
        value = value.detach().cpu().contiguous()
        raw = ctypes.string_at(value.data_ptr(), size) if size else b''
        metadata.append({'name': name, 'dtype': dtype, 'shape': list(value.shape), 'bytes': size})
        chunks.append(raw)
    header = json.dumps(metadata, separators=(',', ':')).encode('ascii')
    if len(header) > 262144 or total + len(header) + 44 > max_bytes:
        raise ValueError('state exceeds byte limit')
    body = _MAGIC + struct.pack('!I', len(header)) + header + b''.join(chunks)
    return body + hashlib.sha256(body).digest()


def decode(payload: bytes, device: str | torch.device = 'cpu', max_bytes: int = _MAX_BYTES) -> State:
    """Validate metadata and byte counts before allocation."""
    if len(payload) > max_bytes or len(payload) < 44 or payload[:8] != _MAGIC:
        raise ValueError('invalid state envelope or byte limit')
    body = memoryview(payload)[:-32]
    if hashlib.sha256(body).digest() != payload[-32:]:
        raise ValueError('state checksum mismatch')
    header_size = struct.unpack('!I', body[8:12])[0]
    if header_size > 262144 or 12 + header_size > len(body):
        raise ValueError('invalid metadata length')
    try:
        metadata = json.loads(bytes(body[12:12 + header_size]))
    except (ValueError, UnicodeError) as error:
        raise ValueError('invalid state metadata') from error
    if not isinstance(metadata, list) or not 1 <= len(metadata) <= 1024:
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
        if not isinstance(shape, list) or len(shape) > 5 or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError('invalid tensor dimensions')
        if type(size) is not int or size != math.prod(shape) * _SIZES[dtype] or offset + size > len(body):
            raise ValueError('invalid tensor byte count')
        entries.append((name, dtype, shape, offset, size))
        seen.add(name)
        offset += size
    if offset != len(body):
        raise ValueError('unexpected trailing bytes')
    state = {}
    for name, dtype, shape, offset, size in entries:
        value = (torch.frombuffer(bytearray(body[offset:offset + size]), dtype=_DTYPES[dtype]).reshape(shape)
                 if size else torch.empty(shape, dtype=_DTYPES[dtype]))
        state[name] = value.to(device)
    return state


def length(state: State) -> int:
    value = state.get('length')
    if not isinstance(value, Tensor) or value.dtype != torch.int64 or value.shape != () or value.item() < 0:
        raise ValueError('invalid cache length')
    return int(value.item())


def digest(tokens: Tensor) -> Tensor:
    raw = encode({'tokens': tokens})
    return torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8, device=tokens.device)


def _tokens(state: State, count: int) -> Tensor:
    value = state.get('tokens')
    if not isinstance(value, Tensor) or value.dtype != torch.int64 or value.ndim not in (1, 2) or value.shape[-1] != count:
        raise ValueError('invalid token identity')
    return value


def delta(state: State, prefix_length: int) -> State:
    """Slice every full-attention KV; always send complete final linear states."""
    total = length(state)
    if type(prefix_length) is not int or not 0 <= prefix_length <= total:
        raise ValueError('invalid destination prefix length')
    tokens = _tokens(state, total)
    change = dict(state)
    change.update(base_length=state['length'].new_tensor(prefix_length),
                  base_digest=digest(tokens[..., :prefix_length]), tokens=tokens[..., prefix_length:])
    for key, value in state.items():
        if key.endswith(('.key', '.value')):
            if value.ndim != 4 or value.shape[-2] != total:
                raise ValueError('invalid full-attention cache')
            change[key] = value[..., prefix_length:, :]
    return change


def merge(base: State | None, change: State) -> State:
    """Reject different model/token prefixes, retaining exact recurrent snapshots."""
    if 'base_length' not in change or 'base_digest' not in change:
        raise ValueError('missing transfer prefix metadata')
    prefix = length({'length': change['base_length']})
    total = length(change)
    if prefix > total or (base is None and prefix) or (base is not None and length(base) != prefix):
        raise ValueError('destination prefix length mismatch')
    suffix = _tokens(change, total - prefix)
    previous = suffix[..., :0] if base is None else _tokens(base, prefix)
    if not torch.equal(digest(previous).cpu(), change['base_digest'].cpu()):
        raise ValueError('destination prefix identity mismatch')
    if base is not None and not torch.equal(base['model_config'].cpu(), change['model_config'].cpu()):
        raise ValueError('different model cache')
    result = {k: v for k, v in change.items() if k not in ('base_length', 'base_digest')}
    result['tokens'] = torch.cat((previous.to(suffix.device), suffix), dim=-1)
    for key, value in change.items():
        if key.endswith(('.key', '.value')):
            if value.ndim != 4 or value.shape[-2] != total - prefix:
                raise ValueError('invalid full-attention suffix')
            old = value[..., :0, :] if base is None else base[key].to(value.device)
            if old.shape[:-2] != value.shape[:-2] or old.shape[-2:] != (prefix, value.shape[-1]):
                raise ValueError('invalid destination full-attention cache')
            result[key] = torch.cat((old, value), dim=-2)
    return result


def state_bytes(state: State) -> int:
    return sum(v.numel() * v.element_size() for k, v in state.items() if k.startswith('layer.'))
