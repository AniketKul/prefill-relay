"""Greedy continuation using one restored native cache for the whole sequence."""
from __future__ import annotations

import torch
from torch import Tensor

from .hf_state import State, length


def _logits(value, expected_shape, device):
    if (not isinstance(value, Tensor) or tuple(value.shape) != expected_shape
            or value.device != device or not value.is_floating_point()):
        raise ValueError('invalid continuation logits shape, device or dtype')
    if not torch.isfinite(value).all():
        raise ValueError('nonfinite continuation logits')
    return value


@torch.inference_mode()
def generate_continuation(backend, state: State, output_tokens: int, *, reserve: bool = False) -> Tensor:
    """Return greedy tokens as ``[batch, output_tokens]`` without changing state.

    The first token comes from cached final logits. The remaining tokens require
    exactly ``output_tokens - 1`` model calls, reusing the returned native cache.
    Model identity and every layer checkpoint are validated by ``_restore`` once.
    Context capacity counts model inputs, including generated tokens fed back in.
    """
    if type(output_tokens) is not int or output_tokens < 1:
        raise ValueError('output_tokens must be a positive integer')
    prefix_length = length(state)
    tokens = state.get('tokens')
    if (not isinstance(tokens, Tensor) or tokens.dtype != torch.int64
            or tokens.ndim not in (1, 2) or tokens.shape[-1] != prefix_length
            or prefix_length < 1 or tokens.device != backend.device):
        raise ValueError('invalid cached token identity')
    batch = 1 if tokens.ndim == 1 else tokens.shape[0]
    if batch < 1:
        raise ValueError('cache batch must be nonempty')
    if prefix_length + output_tokens - 1 > backend.max_context:
        raise ValueError('continuation exceeds model context limit')
    shape = (1, backend.vocab_size) if tokens.ndim == 1 else (batch, 1, backend.vocab_size)
    cached = _logits(state.get('last_logits'), shape, backend.device)
    logits = cached if tokens.ndim == 1 else cached[:, -1]
    past = backend._restore(state, batch, output_tokens - 1) if reserve else backend._restore(state, batch)
    generated = torch.empty((batch, output_tokens), dtype=torch.int64, device=backend.device)
    for index in range(output_tokens):
        token = logits.argmax(dim=-1)
        generated[:, index] = token
        if index + 1 == output_tokens:
            break
        output = backend.model(input_ids=token[:, None], past_key_values=past,
                               use_cache=True, logits_to_keep=1)
        next_past = getattr(output, 'past_key_values', None)
        if next_past is None:
            raise ValueError('model did not return a continuation cache')
        past = next_past
        logits = _logits(getattr(output, 'logits', None),
                         (batch, 1, backend.vocab_size), backend.device)[:, -1]
    return generated
