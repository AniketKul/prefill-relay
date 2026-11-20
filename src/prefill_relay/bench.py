"""Reproducible correctness and arrival-driven workloads on real worker processes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading

from .runtime import Coordinator
from .workload import compare_policies, run_workload, skew_prompts


def validate_service(service,tokens,output_tokens=8):
    service.reset_caches()
    reference=service.reference(tokens,output_tokens)['generated_tokens']
    cold=service.run(tokens,output_tokens,'all-remote')
    assert cold['generated_tokens']==reference, 'independent-process generation differs'
    assert service.remote_host or len(set(cold['worker_pids'].values()))==2, 'prefill and decode share a process'
    # Very short prefills can finish before the sender thread is scheduled.
    assert cold['prefill']['sender']['first_send_at'] is not None, 'no state was transmitted'
    warm=service.run(tokens,output_tokens,'all-remote')
    assert warm['generated_tokens']==reference and warm['transferred_bytes']==0, 'complete-hit reuse failed'
    # Equal-length warmed requests exercise the actual decoder batching queue.
    barrier=threading.Barrier(4)
    def simultaneous(_):
        barrier.wait(); return service.run(tokens,output_tokens)
    with ThreadPoolExecutor(4) as executor: batched=list(executor.map(simultaneous,range(4)))
    assert all(row['generated_tokens']==reference for row in batched), 'batched generation differs'
    service.reset_caches()
    local=service.run(tokens,output_tokens,'all-local')
    assert local['generated_tokens']==reference, 'local generation differs'
    service.reset_caches()
    prefix=max(1,len(tokens)//2)
    service.run(tokens[:prefix],output_tokens,'all-remote')
    resumed=service.run(tokens,output_tokens,'all-remote')
    assert resumed['prefix_tokens']==prefix, 'producer prefix was not reused'
    assert resumed['destination_prefix_tokens']==prefix, 'decoder prefix was not reused'
    assert resumed['generated_tokens']==reference, 'prefix continuation changed generated tokens'
    service.set_bandwidth(0)
    full_hit_outage=service.run(tokens,output_tokens)
    assert full_hit_outage['transferred_bytes']==0, 'full hit used unavailable link'
    service.set_bandwidth(100*2**20)
    service.reset_caches()
    branch=tokens[:16]+[(t+1)%service.workers['remote-0'].info['vocab_size'] for t in tokens[16:]]+[7]*19
    service.run(tokens,output_tokens,'all-remote')
    branched=service.run(branch,output_tokens,'all-remote')
    assert branched['generated_tokens']==service.reference(branch,output_tokens)['generated_tokens'], 'branch changed output'
    stats=service.stats()
    assert all(row['transfers']==0 for row in stats['workers'].values()), 'transfer allocations leaked'
    return dict(reference_tokens=reference,cold=cold,warm=warm,batched=batched,local=local,
                prefix=resumed,full_hit_outage=full_hit_outage,branched=branched,stats=stats)


def validate_migration(service,tokens,output_tokens=4):
    service.reset_caches(); half=max(1,len(tokens)//2)
    service.run(tokens[:half],output_tokens,'all-remote')
    source=('remote','remote-0'); target=('local','local-1')
    initial=service.migrate_prefix(source,target,tokens[:half])
    service.run(tokens,output_tokens,'all-remote')
    delta=service.migrate_prefix(source,target,tokens)
    service.workers[target[1]].call('clear')
    cold=service.migrate_prefix(source,target,tokens)
    assert cold['bytes_transferred']>delta['bytes_transferred']>0, 'migration did not omit cached blocks'
    assert all(r['transfers']==0 for r in service.stats()['workers'].values()), 'migration pins leaked'
    return dict(initial=initial,delta=delta,cold=cold,stats=service.stats())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['tiny','hf'],default='tiny')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--model',default='Qwen/Qwen3.5-0.8B')
    parser.add_argument('--revision',default='main')
    parser.add_argument('--profile',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scenario',choices=['validate','migration','uniform','burst','skew','bandwidth','long','lognormal','suite'],default='validate')
    parser.add_argument('--lengths',type=int,nargs='+',default=[64,256,1024])
    parser.add_argument('--requests',type=int,default=12)
    parser.add_argument('--output-tokens',type=int,default=8)
    parser.add_argument('--rate',type=float,default=2)
    parser.add_argument('--bandwidth-mib',type=float,default=100)
    parser.add_argument('--cache-gib',type=float,default=4)
    parser.add_argument('--block-tokens',type=int,default=16)
    parser.add_argument('--seed',type=int,default=0,help='seed for sampled uncached lengths')
    parser.add_argument('--warm-rounds',type=int,default=0)
    parser.add_argument('--remote-host',help='OpenSSH host alias for the prefill workers')
    parser.add_argument('--remote-project',default='prefill-relay')
    parser.add_argument('--remote-device')
    parser.add_argument('--ssh-streams', type=int, default=1, help='independently authenticated SSH sessions per remote worker')
    parser.add_argument('--remote-cache-dir')
    parser.add_argument('--remote-profile',type=Path,help='prefill timings for the remote hardware')
    parser.add_argument('--local-rdma',action='store_true',help='use host-memory RDMA for local handoff; set PREFILL_RELAY_RDMA_ADDRESS to the local interface IPv4')
    parser.add_argument('--calibrate-tps', type=float, help='cap decode batches to the largest profiled size meeting this per-sequence rate at every profile length')
    parser.add_argument('--remote-workers',type=int,default=1)
    args=parser.parse_args()
    if args.requests<1 or any(n<1 for n in args.lengths): parser.error('positive requests and lengths required')
    if args.backend=='hf':
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,trust_remote_code=False)
        seed=tokenizer('The capital of France is Paris. A reusable prefix saves repeated computation. ').input_ids
    else: seed=[3,20,37,54,71,88,105,122]
    prompts=[(seed*((n+len(seed)-1)//len(seed)))[:n] for n in args.lengths]
    bandwidth=args.bandwidth_mib*2**20
    profile=json.loads(args.profile.read_text()) if args.profile else None
    import os
    local_rdma_address = os.environ.get('PREFILL_RELAY_RDMA_ADDRESS') if args.local_rdma else None
    if args.local_rdma and not local_rdma_address:
        parser.error('--local-rdma requires PREFILL_RELAY_RDMA_ADDRESS')
    with tempfile.TemporaryDirectory(prefix='prefill-relay-') as cache, Coordinator(cache,
            remote_host=args.remote_host,remote_project=args.remote_project,remote_device=args.remote_device,ssh_streams=args.ssh_streams,
            remote_cache_dir=args.remote_cache_dir,remote_profile=json.loads(args.remote_profile.read_text()) if args.remote_profile else None,
            local_rdma_address=local_rdma_address,decode_slo_tps=args.calibrate_tps,backend=args.backend,device=args.device,model_id=args.model,revision=args.revision,
            bandwidth_bytes_s=bandwidth,profile=profile,output_tokens=args.output_tokens,
            cache_bytes=int(args.cache_gib*2**30),remote_workers=args.remote_workers,batch_wait_s=.02 if args.scenario=='validate' else .005,block_tokens=args.block_tokens) as service:
        result={'kind':('physical SSH prefill host with local decoding' if args.remote_host else 'independent model processes on one host with application-rate emulation'),
                'workers':{n:{k:w.info[k] for k in ('model_id','revision','pid','kernels')} for n,w in service.workers.items()},
                'configuration':{**{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('remote_host','remote_project','remote_cache_dir')},'batch_wait_s':service.config['batch_wait_s']}}
        if args.scenario=='validate': result['validation']=validate_service(service,prompts[0],args.output_tokens)
        elif args.scenario=='migration': result['migration']=validate_migration(service,prompts[0],args.output_tokens)
        elif args.scenario=='long':
            rows=[]
            for tokens in prompts:
                service.reset_caches()
                row=service.run(tokens,args.output_tokens,'all-remote')
                row['prompt_tokens']=len(tokens)
                row['reference_equal']=row['generated_tokens']==service.reference(tokens,args.output_tokens)['generated_tokens']
                assert row['reference_equal'], 'long-context generation differs'
                rows.append(row)
                print(json.dumps({'completed_context':len(tokens),'ttft_s':row['ttft_s'],'bytes':row['transferred_bytes']}),flush=True)
            result['long_context']=rows
        else:
            if args.scenario == 'lognormal':
                from .analytics import Workload
                lengths = [round(n) for n in Workload().sample(args.requests, args.seed)]
                prompts = [(seed*((n+len(seed)-1)//len(seed)))[:n] for n in lengths]
                result['sampled_lengths'] = lengths
                result['length_distribution'] = dict(mu=9.90,sigma=1.,low=128,high=128000,seed=args.seed)
                # Warm a representative shape; exact-shape compilation remains in request times.
                service.warmup([seed])
            else:
                service.warmup(prompts)
            result['compute_warmup']='all workers, each full prompt shape, prefill and decode batches1/2/4; caches reset afterwards'
            if args.scenario == 'lognormal':
                result['compute_warmup'] = 'all workers, seed prompt only; measured lengths may include compilation'
            scenarios=['uniform','burst','skew','bandwidth','warm'] if args.scenario=='suite' else [args.scenario]
            runs={}
            for scenario in scenarios:
                requests=skew_prompts(prompts,args.requests) if scenario=='skew' else [
                    [*prompts[i%len(prompts)][:-1],(i+11)%next(iter(service.workers.values())).info['vocab_size']]
                    for i in range(args.requests)]
                options=dict(output_tokens=args.output_tokens,rate=args.rate,max_workers=8,
                             warm_rounds=1 if scenario=='warm' else args.warm_rounds,initial_bandwidth_bytes_s=bandwidth)
                if scenario=='burst': options['pattern']='burst'
                if scenario=='bandwidth': options['bandwidth_changes']=[((args.requests-1)/args.rate/2,bandwidth/20)]
                runs[scenario]=compare_policies(service,requests,**options)
                print(json.dumps({'scenario':scenario,'summaries':{p:r['summary'] for p,r in runs[scenario].items()}}),flush=True)
            result['workloads']=runs
            if args.scenario!='suite': result['policies']=runs[args.scenario]
            result['passed']=all(r['summary']['errors']==0 for policies in runs.values() for r in policies.values())
        result['final_stats']=service.stats()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'scenario':args.scenario,'completed':True,'passed':result.get('passed',True)}))
    if not result.get('passed',True): raise RuntimeError('workload requests failed; inspect saved request errors')


if __name__=='__main__': main()
