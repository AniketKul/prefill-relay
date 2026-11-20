"""Request-owned full-attention cache for a known continuation length."""
from transformers.cache_utils import DynamicLayer


class DecodeLayer(DynamicLayer):
    """Keep the native logical length and mask while appending into reserved storage."""

    def __init__(self, keys, values, steps):
        super().__init__()
        length = keys.shape[-2]
        shape = (*keys.shape[:-2], length + steps, keys.shape[-1])
        self.key_storage = keys.new_empty(shape)
        self.value_storage = values.new_empty(shape)
        self.keys = self.key_storage[..., :length, :]
        self.values = self.value_storage[..., :length, :]
        self.keys.copy_(keys)
        self.values.copy_(values)
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        start = self.keys.shape[-2]
        end = start + key_states.shape[-2]
        if end > self.key_storage.shape[-2]:
            raise ValueError('decode cache capacity exceeded')
        self.key_storage[..., start:end, :].copy_(key_states)
        self.value_storage[..., start:end, :].copy_(value_states)
        self.keys = self.key_storage[..., :end, :]
        self.values = self.value_storage[..., :end, :]
        return self.keys, self.values
