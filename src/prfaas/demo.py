"""Real tensor prefill/decode and TCP handoff; a correctness demo, not a serving benchmark."""
import argparse
import json
import time
from uuid import uuid4
import torch
from .analytics import count
from .cache import HybridCache, CacheDirectory
from .scheduler import Router
from .tiny_model import TinyHybrid, encode_state, decode_state, delta, merge
from .transport import Receiver, StreamSender


def new_pool():
    return HybridCache(block_tokens=16,block_bytes=16384,capacity_blocks=4096,namespace='tiny-hybrid-v1-seed0')


def cache_state(pool,state):
    n=int(state['length'])
    if n%pool.block_tokens: return
    full=[]
    for i in range(0,n,pool.block_tokens):
        full.append(encode_state({k:state[k][i:i+pool.block_tokens] for k in ('layer.3.key','layer.3.value')}))
    recurrent={str(i):encode_state({k:v for k,v in state.items() if k.startswith(f'layer.{i}.')}) for i in range(3)}
    pool.put(state['tokens'].tolist(),{'attention':tuple(full)},recurrent)


def lookup(pool,tokens,model):
    hit=pool.match(tokens.tolist())
    if not hit.length: return None
    device=tokens.device
    state={'length':torch.tensor(hit.length,device=device),'tokens':tokens[:hit.length], 'model_config':model.config.clone()}
    pieces=[decode_state(block,device) for block in hit.full['attention']]
    for key in ('layer.3.key','layer.3.value'): state[key]=torch.cat([s[key] for s in pieces])
    for payload in hit.recurrent.values(): state.update(decode_state(payload,device))
    return state


def _sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize()


def run_case(model,tokens,decode_tokens,local_prefix,remote_prefix,threshold,abundant,connections):
    device=tokens.device
    local,remote=new_pool(),new_pool()
    for pool,n in ((local,local_prefix),(remote,remote_prefix)):
        if n:
            _,state=model(tokens[:n]); cache_state(pool,state)
    directory=CacheDirectory(); directory.register('local','pd-p',local); directory.register('remote','prfaas',remote)
    _,lpd=directory.best('local',local.namespace,tokens.tolist())
    _,lremote=directory.best('remote',remote.namespace,tokens.tolist())
    placement=Router(threshold).route(len(tokens),lpd,lremote,abundant)
    pools={'local':local,'remote':remote}
    source=lookup(pools[placement.prefix_source],tokens,model)
    destination=lookup(local,tokens,model)
    destination_length=0 if destination is None else int(destination['length'])
    moved=0; layer_count=0; sender_stats={}; _sync(device); start=time.perf_counter()
    with Receiver() as receiver:
        if placement.migrate_prefix:
            request=uuid4().hex
            target_base=lookup(pools[placement.target],tokens,model)
            base_length=0 if target_base is None else int(target_base['length'])
            payload=encode_state(delta(source,base_length))
            with StreamSender(receiver.address,request,connections,pool=pools[placement.prefix_source]) as sender:
                sender.submit('prefix',payload)
            moved+=sender.stats['bytes']
            source=merge(target_base,decode_state(receiver.take(request)['prefix'],device))
            cache_state(pools[placement.target],source)
        suffix=tokens[placement.prefix_tokens:]
        if not len(suffix): raise ValueError('demo requires at least one uncached token')
        if placement.target=='remote':
            request=uuid4().hex
            with StreamSender(receiver.address,request,connections,pool=remote) as sender:
                def on_layer(index,state):
                    sender.submit(str(index),encode_state(delta(state,destination_length)))
                logits,state=model(suffix,source,on_layer)
            frames=receiver.take(request)
            if set(frames)!={'0','1','2','3'}: raise RuntimeError('incomplete layer transfer')
            changes={}
            for index in range(4): changes.update(decode_state(frames[str(index)],device))
            decode_state_value=merge(destination,changes)
            sender_stats=sender.stats; moved+=sender.stats['bytes']; layer_count=len(frames)
        else:
            logits,state=model(suffix,source)
            decode_state_value=state
        _sync(device); prefill_handoff_s=time.perf_counter()-start
        cache_state(pools[placement.target],state); cache_state(local,decode_state_value)
    expected,_=model(tokens)
    torch.testing.assert_close(logits[-1],expected[-1],rtol=3e-5,atol=3e-6)
    error=float((logits[-1]-expected[-1]).abs().max())
    generated=[]; all_tokens=tokens; next_logits=logits[-1]
    for _ in range(decode_tokens):
        token=next_logits.argmax().reshape(1)
        generated.append(int(token)); all_tokens=torch.cat((all_tokens,token))
        incremental,decode_state_value=model(token,decode_state_value)
        reference,_=model(all_tokens)
        torch.testing.assert_close(incremental[-1],reference[-1],rtol=3e-5,atol=3e-6)
        if int(incremental[-1].argmax())!=int(reference[-1].argmax()): raise AssertionError('generated token differs')
        error=max(error,float((incremental[-1]-reference[-1]).abs().max()))
        next_logits=incremental[-1]
    if remote._transfers or local._transfers: raise AssertionError('transfer blocks leaked')
    return {'target':placement.target,'prefix_source':placement.prefix_source,'prefix_tokens':placement.prefix_tokens,
            'destination_prefix_tokens':destination_length,'uncached_tokens':placement.uncached_tokens,
            'migrated_prefix':placement.migrate_prefix,'transferred_bytes':moved,'layers_streamed':layer_count,
            'prefill_handoff_s':prefill_handoff_s,'max_abs_logit_error':error,'generated_tokens':generated,
            'sender':sender_stats,'correct':True}


def run_demo(device='cpu',prompt_tokens=512,prefix_tokens=128,decode_tokens=8,connections=4):
    for v,k in [(prompt_tokens,'prompt'),(prefix_tokens,'prefix'),(decode_tokens,'decode'),(connections,'connections')]: count(v,k,zero=False)
    if prefix_tokens%16 or prompt_tokens%16 or prefix_tokens*2>=prompt_tokens or prompt_tokens+decode_tokens>8192:
        raise ValueError('prompt/prefix must be multiples of 16, prompt > 2*prefix, total <=8192')
    model=TinyHybrid().to(device).eval()
    tokens=(torch.arange(prompt_tokens,device=device,dtype=torch.int64)*17+3)%model.vocab_size
    model(tokens[:32]); _sync(device)
    cases={}
    cases['cold_remote']=run_case(model,tokens,decode_tokens,0,0,prefix_tokens,False,connections)
    cases['remote_has_longer_prefix']=run_case(model,tokens,decode_tokens,prefix_tokens,2*prefix_tokens,prefix_tokens,False,connections)
    short=tokens[:2*prefix_tokens+16]
    cases['abundant_prefix_migration']=run_case(model,short,decode_tokens,prefix_tokens,2*prefix_tokens,32,True,connections)
    cases['short_local']=run_case(model,short,decode_tokens,2*prefix_tokens,0,32,False,connections)
    return {'kind':'measured correctness demo; untrained tiny hybrid; single host TCP loopback',
            'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name() if str(device).startswith('cuda') else str(device),
            'prompt_tokens':prompt_tokens,'decode_tokens':decode_tokens,'connections':connections,'cases':cases}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu'); p.add_argument('--prompt-tokens',type=int,default=512)
    p.add_argument('--prefix-tokens',type=int,default=128); p.add_argument('--decode-tokens',type=int,default=8)
    p.add_argument('--connections',type=int,default=4)
    args=p.parse_args()
    print(json.dumps(run_demo(**vars(args)),indent=2))


if __name__=='__main__': main()
