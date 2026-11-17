"""Reproducible pretrained GPU cache-handoff validation."""
from __future__ import annotations

import time
import uuid

import torch

from .hf_state import state_bytes
from .transport import Receiver, StreamSender


def validate_backend(backend, output_tokens: int = 8) -> dict:
    """Run real model, tensor wire, layer pipeline, prefix and batch oracles."""
    if type(output_tokens) is not int or output_tokens < 1:
        raise ValueError('output_tokens must be positive')
    text = 'The capital of France is Paris. The capital of Japan is'
    tokens = backend.tokenizer(text, return_tensors='pt').input_ids[0].to(backend.device)
    events = []
    request_id = uuid.uuid4().hex
    start = time.perf_counter()
    with Receiver(max_bytes=512 * 1024**2, timeout=60) as receiver:
        with StreamSender(receiver.address, request_id, timeout=60) as sender:
            def transfer(index, state):
                events.append({'layer': index, 'seconds': time.perf_counter() - start})
                sender.submit(str(index), backend.encode(state))
            logits, local = backend.forward(tokens, on_layer=transfer)
        transferred = {}
        for payload in receiver.take(request_id).values():
            transferred.update(backend.decode(payload))
    assert len(events) == backend.num_layers, 'missing layer callback'
    assert events[0]['seconds'] < events[-1]['seconds'], 'layers did not stream progressively'
    assert set(local) == set(transferred), 'incomplete transported state'
    assert all(torch.equal(local[k], transferred[k]) for k in local), 'wire changed state'
    prompt_state = local
    direct, remote, generated, max_error = local, transferred, [], 0.0
    next_token = logits[-1].argmax().reshape(1)
    for _ in range(output_tokens):
        generated.append(int(next_token.item()))
        a, direct = backend.forward(next_token, direct)
        b, remote = backend.forward(next_token, remote)
        max_error = max(max_error, float((a - b).abs().max()))
        assert torch.equal(a, b), 'handoff changed decode logits'
        next_token = a[-1].argmax().reshape(1)
    prefix_length = max(4, tokens.numel() // 2)
    _, prefix = backend.forward(tokens[:prefix_length])
    prefix_before = {k: v.clone() for k, v in prefix.items()}
    resumed_logits, resumed = backend.forward(tokens[prefix_length:], prefix)
    assert all(torch.equal(prefix[k], prefix_before[k]) for k in prefix), 'prefix reuse mutated the checkpoint'
    prefix_error = float((logits - resumed_logits).abs().max())
    assert logits[-1].argmax() == resumed_logits[-1].argmax(), 'prefix continuation changed next token'
    change = backend.delta(resumed, prefix_length)
    merged = backend.merge(prefix, backend.decode(backend.encode(change)))
    assert all(torch.equal(merged[k], resumed[k]) for k in resumed), 'delta changed producer state'
    # Full hits must work without invoking the model at all.
    original_forward = backend.model.forward
    def reject_forward(*args, **kwargs):
        raise AssertionError('complete cache hit invoked model')
    backend.model.forward = reject_forward
    try:
        full_hit_layers = []
        cached_logits, _ = backend.prefill(tokens[:0], prompt_state, on_layer=lambda i, s: full_hit_layers.append(i))
        assert full_hit_layers == list(range(backend.num_layers)), 'full hit did not emit cached layers'
        assert torch.equal(cached_logits, logits), 'cached final logits changed'
    finally:
        backend.model.forward = original_forward
    # Simulate a shorter decode prefix than the prefix available at prefill.
    destination_length = max(1, prefix_length // 2)
    _, destination = backend.forward(tokens[:destination_length])
    missing = backend.delta(resumed, destination_length)
    rebuilt = backend.merge(destination, backend.decode(backend.encode(missing)))
    assert rebuilt['length'].item() == tokens.numel()
    for name, tensor in missing.items():
        if name.endswith(('.key', '.value')):
            assert tensor.shape[-2] == tokens.numel() - destination_length
    batch_tokens = tokens.unsqueeze(0).repeat(2, 1)
    batch_logits, batch_state = backend.forward(batch_tokens)
    assert batch_logits.shape == (2, 1, backend.vocab_size)
    assert batch_logits[0, -1].argmax() == logits[-1].argmax()
    batched_next = batch_logits[:, -1].argmax(-1, keepdim=True)
    batch_a, _ = backend.forward(batched_next, batch_state)
    batch_b, _ = backend.forward(batched_next, backend.decode(backend.encode(batch_state)))
    assert torch.equal(batch_a, batch_b), 'batched handoff changed decode logits'
    return {'model': backend.model_id, 'revision': backend.revision, 'dtype': str(backend.model.dtype),
            'device': str(backend.device), 'prompt_tokens': tokens.numel(), 'decode_tokens': generated,
            'completion': backend.tokenizer.decode(generated), 'layers_streamed': len(events),
            'first_layer_ready_seconds': events[0]['seconds'], 'last_layer_ready_seconds': events[-1]['seconds'],
            'handoff_max_logit_error': max_error, 'prefix_max_logit_error': prefix_error,
            'prefix_next_token_equal': True, 'full_hit_model_calls': 0, 'batch_size_validated': 2,
            'wire_state_equal': True, 'delta_state_equal': True, 'cache_bytes': state_bytes(prompt_state),
            'unequal_prefix_missing_tokens': tokens.numel() - destination_length}
