"""Pretrained text inference with transferable hybrid attention checkpoints."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable

import torch
from torch import Tensor

from . import hf_state
from .hf_state import State


class HFHybrid:
    """Qwen3.5 text adapter; equal-length batches and exact prefix continuation.

    Only final-token logits are materialized. Each layer callback receives an
    independent cache snapshot and full request/model identity metadata.
    The adapter is intentionally pinned to the tested Transformers cache API.
    """

    encode = staticmethod(hf_state.encode)
    delta = staticmethod(hf_state.delta)
    merge = staticmethod(hf_state.merge)

    def __init__(self, model_id: str = 'Qwen/Qwen3.5-0.8B', revision: str = 'main',
                 device: str = 'cuda', dtype: str = 'bfloat16'):
        if str(device).startswith('cuda'):
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # fail explicitly if GPU kernels cannot import
            try: importlib.metadata.version('causal-conv1d')
            except importlib.metadata.PackageNotFoundError: pass
            else: __import__('causal_conv1d_cuda')
        from huggingface_hub import model_info
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM

        if dtype not in ('float32', 'float16', 'bfloat16'):
            raise ValueError('dtype must be float32, float16 or bfloat16')
        self.model_id = model_id
        self.revision = model_info(model_id, revision=revision).sha
        self.model = Qwen3_5ForCausalLM.from_pretrained(
            model_id, revision=self.revision, dtype=getattr(torch, dtype),
            attn_implementation='sdpa', trust_remote_code=False).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=self.revision, trust_remote_code=False)
        self._initialize(dtype)

    def _initialize(self, dtype: str):
        self.config = self.model.config
        if self.config.model_type != 'qwen3_5_text':
            raise ValueError('HFHybrid supports Qwen3.5 text checkpoints')
        self.device = next(self.model.parameters()).device
        self.vocab_size = self.config.vocab_size
        self.num_layers = self.config.num_hidden_layers
        self.kv_sequence_dim = 2
        self.max_context = self.config.max_position_embeddings
        from transformers.models.qwen3_5 import modeling_qwen3_5 as kernels
        try: convolution_version = importlib.metadata.version('causal-conv1d')
        except importlib.metadata.PackageNotFoundError: convolution_version = None
        self.kernel_provenance = {'causal_conv1d_version': convolution_version,
            'prefill_convolution': getattr(kernels.causal_conv1d_fn, '__module__', 'torch_reference'),
            'decode_convolution': getattr(kernels.causal_conv1d_update, '__module__', 'torch_reference')}
        identity = {'model': self.model_id, 'revision': self.revision, 'dtype': dtype,
                    'config': self.config.to_dict(), 'adapter': 1}
        raw = json.dumps(identity, sort_keys=True, default=str).encode()
        self.identity = torch.tensor(list(hashlib.sha256(raw).digest()), dtype=torch.uint8, device=self.device)

    def decode(self, payload: bytes, device=None, **kwargs) -> State:
        return hf_state.decode(payload, self.device if device is None else device, **kwargs)

    def _layer(self, cache, index: int, clone: bool = False) -> State:
        layer, result = cache.layers[index], {}
        if self.config.layer_types[index] == 'full_attention':
            values = {'key': layer.keys, 'value': layer.values}
        else:
            values = {'conv': layer.conv_states[0], 'recurrent': layer.recurrent_states[0]}
        for name, value in values.items():
            if not isinstance(value, Tensor):
                raise ValueError(f'missing layer {index} {name} state')
            result[f'layer.{index}.{name}'] = value.clone() if clone else value
        return result

    def _restore(self, state: State | None, batch: int):
        from transformers import DynamicCache

        cache = DynamicCache(config=self.config)
        if state is None:
            return cache
        total = hf_state.length(state)
        if not torch.equal(state.get('model_config', torch.empty(0)).cpu(), self.identity.cpu()):
            raise ValueError('different model cache')
        expected = {'length', 'tokens', 'model_config', 'last_logits'}
        tokens = hf_state._tokens(state, total)
        if (1 if tokens.ndim == 1 else tokens.shape[0]) != batch:
            raise ValueError('cache batch mismatch')
        for i, kind in enumerate(self.config.layer_types):
            names = ('key', 'value') if kind == 'full_attention' else ('conv', 'recurrent')
            expected.update(f'layer.{i}.{name}' for name in names)
            values = [state.get(f'layer.{i}.{name}') for name in names]
            if any(not isinstance(v, Tensor) or v.device != self.device or not v.is_floating_point()
                   or v.ndim == 0 or v.shape[0] != batch for v in values):
                raise ValueError('invalid layer cache tensors')
            if kind == 'full_attention':
                shape = (batch, self.config.num_key_value_heads, total, self.config.head_dim)
                if any(tuple(v.shape) != shape or v.dtype != self.model.dtype for v in values):
                    raise ValueError('invalid full-attention cache shape or dtype')
                cache.update(values[0].clone(), values[1].clone(), i)
            else:
                cfg = self.config
                channels = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + cfg.linear_num_value_heads * cfg.linear_value_head_dim
                conv_shape = (batch, channels, cfg.linear_conv_kernel_dim)
                recurrent_shape = (batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
                if tuple(values[0].shape) != conv_shape or tuple(values[1].shape) != recurrent_shape:
                    raise ValueError('invalid linear-attention cache shape')
                if values[0].dtype != self.model.dtype or values[1].dtype != torch.float32:
                    raise ValueError('invalid linear-attention cache dtype')
                cache.update_conv_state(values[0], i, conv_kernel_size=cfg.linear_conv_kernel_dim)
                cache.update_recurrent_state(values[1], i)
        if set(state) != expected:
            raise ValueError('incomplete or unexpected cache keys')
        return cache

    @torch.inference_mode()
    def forward(self, tokens: Tensor, cache: State | None = None,
                on_layer: Callable[[int, State], None] | None = None) -> tuple[Tensor, State]:
        if not isinstance(tokens, Tensor) or tokens.dtype != torch.int64 or tokens.ndim not in (1, 2):
            raise ValueError('tokens must be an int64 vector or equal-length batch')
        if tokens.device != self.device or (tokens < 0).any() or (tokens >= self.vocab_size).any():
            raise ValueError('invalid token id or device')
        batch = 1 if tokens.ndim == 1 else tokens.shape[0]
        start = 0 if cache is None else hf_state.length(cache)
        total = start + tokens.shape[-1]
        if total > self.max_context:
            raise ValueError('input exceeds model context limit')
        past = self._restore(cache, batch)
        if not tokens.numel():
            if cache is None:
                raise ValueError('empty input requires a complete cache')
            logits = cache.get('last_logits')
            shape = (1, self.vocab_size) if tokens.ndim == 1 else (batch, 1, self.vocab_size)
            if not isinstance(logits, Tensor) or tuple(logits.shape) != shape:
                raise ValueError('invalid cached logits')
            if on_layer is not None:
                metadata = {name: cache[name] for name in ('length', 'tokens', 'model_config')}
                for i in range(self.num_layers):
                    layer_state = {**metadata, **self._layer(past, i, clone=True)}
                    if i == self.num_layers - 1:
                        layer_state['last_logits'] = logits.clone()
                    on_layer(i, layer_state)
            return logits, cache
        all_tokens = tokens.clone() if cache is None else torch.cat((cache['tokens'], tokens), dim=-1)
        metadata = {'length': tokens.new_tensor(total), 'tokens': all_tokens, 'model_config': self.identity.clone()}
        hooks = []
        if on_layer is not None:
            for i, layer in enumerate(self.model.model.layers):
                def hook(_module, _args, _output, index=i):
                    if index != len(self.config.layer_types) - 1:
                        on_layer(index, {**metadata, **self._layer(past, index, clone=True)})
                hooks.append(layer.register_forward_hook(hook))
        try:
            output = self.model(input_ids=tokens.unsqueeze(0) if tokens.ndim == 1 else tokens,
                                past_key_values=past, use_cache=True, logits_to_keep=1)
        finally:
            for hook in hooks:
                hook.remove()
        state = dict(metadata)
        for i in range(len(self.config.layer_types)):
            state.update(self._layer(output.past_key_values, i))
        logits = output.logits[0] if tokens.ndim == 1 else output.logits
        state['last_logits'] = logits.clone()
        if on_layer is not None:
            index = len(self.config.layer_types) - 1
            on_layer(index, {**metadata, **self._layer(past, index, clone=True), 'last_logits': state['last_logits']})
        return logits, state

    def enable_decode_graphs(self):
        from .mlp_graphs import DecodeMLPGraphs
        if getattr(self, '_decode_graphs', None) is None:
            self._decode_graphs = DecodeMLPGraphs(self)

    def generate(self, state: State, output_tokens: int) -> Tensor:
        from .hf_decode import generate_continuation
        graphs = getattr(self, '_decode_graphs', None)
        if graphs is not None:
            with graphs.enabled():
                return generate_continuation(self, state, output_tokens)
        return generate_continuation(self, state, output_tokens)

    __call__ = forward

    @torch.inference_mode()
    def prefill(self, tokens: Tensor, cache: State | None = None, chunk_size: int = 4096,
                on_layer: Callable[[int, State], None] | None = None) -> tuple[Tensor, State]:
        """Bound peak activation memory; callbacks run on each final-layer checkpoint.

        Only the final chunk emits callbacks, so all layers describe the same
        complete token prefix and can immediately be handed to a decode worker.
        """
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        if not isinstance(tokens, Tensor) or tokens.ndim not in (1, 2):
            raise ValueError('tokens must be a vector or equal-length batch')
        if not tokens.numel():
            return self.forward(tokens, cache, on_layer)
        for start in range(0, tokens.shape[-1], chunk_size):
            end = min(start + chunk_size, tokens.shape[-1])
            logits, cache = self.forward(tokens[..., start:end], cache,
                                         on_layer if end == tokens.shape[-1] else None)
        return logits, cache
