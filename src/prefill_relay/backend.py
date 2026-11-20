"""Backend selection and block storage for hybrid inference state."""
import torch
from .tiny_model import TinyHybrid, encode_state, decode_state, delta, merge


class TinyBackend:
    def __init__(self, device='cpu'):
        self.model=TinyHybrid().to(device).eval(); self.device=next(self.model.parameters()).device
        self.vocab_size=self.model.vocab_size; self.layers=4; self.sequence_axis=0
        self.model_id='tiny-hybrid'; self.revision='seed0-width32'

    encode=staticmethod(encode_state)
    delta=staticmethod(delta)
    merge=staticmethod(merge)

    def decode(self,payload): return decode_state(payload,self.device)

    def forward(self,tokens,cache=None,on_layer=None):
        if not tokens.numel():
            if cache is None or 'last_logits' not in cache: raise ValueError('missing full-hit logits')
            if on_layer:
                for i in range(self.layers):
                    on_layer(i,{k:v for k,v in cache.items() if not k.startswith('layer.') or k.startswith(f'layer.{i}.')})
            return cache['last_logits'],cache
        original=None if cache is None else {k:v for k,v in cache.items() if k!='last_logits'}
        last=[]
        def hook(i,state):
            if i==self.layers-1: last.append(state)
            elif on_layer: on_layer(i,state)
        logits,state=self.model(tokens,original,hook)
        logits=logits[-1:]; state['last_logits']=logits
        if on_layer: on_layer(self.layers-1,{**last[0],'last_logits':logits})
        return logits,state

    prefill=forward


def load_backend(kind='tiny',device='cpu',model_id=None,revision='main'):
    if kind=='tiny': return TinyBackend(device)
    if kind!='hf': raise ValueError('backend must be tiny or hf')
    from .hf_backend import HFHybrid
    backend=HFHybrid(model_id=model_id or 'Qwen/Qwen3.5-0.8B',revision=revision,device=device)
    backend.layers=backend.num_layers
    backend.sequence_axis=backend.kv_sequence_dim
    return backend


def store_state(pool,backend,state):
    n=int(state['length'])
    keys=[k for k in state if k.endswith(('.key','.value'))]
    full={}
    for key in keys:
        blocks=[]
        for i in range(0,n,pool.block_tokens):
            block=state[key].narrow(backend.sequence_axis,i,min(pool.block_tokens,n-i))
            blocks.append(backend.encode({key:block}))
        full[key]=tuple(blocks)
    recurrent={'checkpoint':backend.encode({k:v for k,v in state.items() if k not in keys})}
    try: pool.put(state['tokens'].tolist(),full,recurrent)
    except MemoryError: return False
    except ValueError as error:
        if str(error) in ('conflicting state for the same model and prefix','checkpoint group schema conflict') and pool.match_length(state['tokens'].tolist())==n:
            return False
        raise
    return True


def restore_state(pool,backend,tokens):
    hit=pool.match(tokens)
    if not hit.length: return None
    state=backend.decode(hit.recurrent['checkpoint'])
    for key,blocks in hit.full.items():
        state[key]=torch.cat([backend.decode(b)[key] for b in blocks],dim=backend.sequence_axis)
    if state['tokens'].tolist()!=tokens[:hit.length]: raise ValueError('cache token identity mismatch')
    return state


def prefix_fingerprints(backend, state, length):
    """Hash only full-attention bytes which the receiver would keep."""
    import hashlib
    if not length or state is None:
        return {}
    return {key: hashlib.sha256(backend.encode({key: value.narrow(
                backend.sequence_axis, 0, length)})).hexdigest()
            for key, value in state.items() if key.endswith(('.key', '.value'))}


def handoff_delta(backend, layer, prefix, fingerprints):
    change = backend.delta(layer, prefix)
    for key, digest in prefix_fingerprints(backend, layer, prefix).items():
        if fingerprints.get(key) != digest:
            # Equal token IDs can produce different bf16 KV on a different prompt shape.
            change['replacement.' + key] = layer[key].narrow(backend.sequence_axis, 0, prefix)
    return change


def merge_handoff(backend, base, change):
    state = dict(base) if base is not None else None
    for name in list(change):
        if name.startswith('replacement.'):
            key = name.removeprefix('replacement.')
            if state is None or key not in state or not key.endswith(('.key', '.value')):
                raise ValueError('unknown replacement tensor')
            value = change.pop(name)
            if value.shape != state[key].shape or value.dtype != state[key].dtype:
                raise ValueError('incompatible replacement tensor')
            state[key] = value
    return backend.merge(state, change)
