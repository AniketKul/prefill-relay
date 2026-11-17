"""Commands: search CONFIG, demo (optional PyTorch)."""
import argparse
import csv
import json
from pathlib import Path
from .analytics import Profile, Workload, grid_search


def search(config):
    if not isinstance(config.get('provenance'),str) or not config['provenance'].strip():
        raise ValueError('config must identify profiling provenance')
    w=Workload(**config['workload'])
    remote=Profile(**config['remote_profile']); local=Profile(**config['local_profile'])
    settings=(config['bandwidth_gbps']*1e9/8,config['batch_max'],config['decode_step_s'],config['output_tokens'])
    best,rows=grid_search(w,remote,local,config['remote_instances'],config['local_instances'],*settings,config['thresholds'])
    homogeneous,_=grid_search(w,remote,local,0,config['homogeneous_instances'],*settings,[w.high])
    naive,_=grid_search(w,remote,local,config['remote_instances'],config['local_instances'],*settings,[0],prefill_counts=[0])
    return {'kind':'analytical capacity from supplied profiles; not measured TTFT','provenance':config['provenance'],
            'prfaas':best,'homogeneous':homogeneous,'naive_heterogeneous':naive},rows


def write_sweep_svg(path,rows,best):
    """Capacity cross-sections from the actual supplied-profile search."""
    threshold_rows=[r for r in rows if r['prefill_instances']==best['prefill_instances']]
    allocation_rows=[r for r in rows if r['threshold']==best['threshold']]
    panels=[('Local prefill allocation','prefill_instances',allocation_rows),('Routing threshold (tokens)','threshold',threshold_rows)]
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="380" viewBox="0 0 1000 380">',
           '<rect width="1000" height="380" fill="white"/>','<text x="30" y="25" font-family="sans-serif" font-size="16">Analytical capacity from supplied profiles</text>']
    for panel,(title,key,data) in enumerate(panels):
        left=60+panel*490; top=65; width=400; height=230
        xmax=max(r[key] for r in data) or 1; ymax=max(r['throughput'] for r in data) or 1
        parts.append(f'<path d="M {left} {top} V {top+height} H {left+width}" fill="none" stroke="black"/>')
        points=' '.join(f"{left+r[key]/xmax*width:.2f},{top+height-r['throughput']/ymax*height:.2f}" for r in data)
        parts.append(f'<polyline points="{points}" fill="none" stroke="#147d92" stroke-width="2"/>')
        for text,x,y in [(title,left,340),('req/s',left-40,top-10),(f'{ymax:.3f}',left-45,top+8),('0',left-12,top+height+15),(f'{xmax:g}',left+width-30,top+height+20)]:
            parts.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="12">{text}</text>')
    parts.append('</svg>'); path.write_text('\n'.join(parts))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('search'); p.add_argument('config',type=Path); p.add_argument('--output',type=Path)
    sub.add_parser('demo',add_help=False)
    args,extra=parser.parse_known_args()
    if args.command=='demo':
        from .demo import main as demo_main
        import sys
        sys.argv=[sys.argv[0],*extra]; demo_main(); return
    if extra: parser.error('unrecognized arguments: '+' '.join(extra))
    result,rows=search(json.loads(args.config.read_text()))
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        with (args.output/'sweep.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write_sweep_svg(args.output/'sweep.svg',rows,result['prfaas'])
        (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
