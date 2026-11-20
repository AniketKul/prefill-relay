"""Independent model worker with loopback control and streaming cache reception."""
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import hmac
import json
import multiprocessing
from pathlib import Path
import threading
import time
from urllib.request import Request, urlopen

from .analytics import count


class WorkerError(RuntimeError):
    def __init__(self,message,transport_stats=None):
        super().__init__(message); self.transport_stats=transport_stats


def call(url,token,operation,**data):
    request=Request(url,data=json.dumps({'operation':operation,**data}).encode(),
                    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urlopen(request,timeout=900) as response:
        result=json.load(response)
    if 'error' in result: raise WorkerError(result['error'],result.get('transport_stats'))
    return result


class ModelWorker:
    def __init__(self,config,stack):
        import torch
        from .backend import load_backend
        from .storage import PersistentHybridCache
        from .cache_service import CacheNode
        from .transport import Receiver
        self.torch=torch; torch.set_num_threads(config.get('cpu_threads',2))
        self.backend=load_backend(config['backend'],config['device'],config.get('model_id'),config.get('revision','main'))
        self.token=config['token']; self.role=config['role']; self.pending={}; self.lock=threading.RLock()
        model=self.backend
        identity=getattr(model,'identity',getattr(getattr(model,'model',None),'config',None))
        namespace=hashlib.sha256(model.encode({'identity':identity})).hexdigest()
        self.pool=PersistentHybridCache(config['cache_path'],namespace=namespace,
            block_tokens=config.get('block_tokens',16),block_bytes=4096,
            capacity_blocks=config.get('cache_bytes',4*2**30)//4096,ram_bytes=config.get('ram_bytes',64*2**20))
        stack.callback(self.pool.close)
        from .rate_limit import rate_client
        self.rate_limiter=lambda size:rate_client(config['rate_url'],self.token,size)
        self.node=stack.enter_context(CacheNode(self.pool,token=self.token,max_body_bytes=config.get('cache_bytes',4*2**30)*2,timeout=120,rate_limiter=self.rate_limiter))
        self.receiver=stack.enter_context(Receiver(max_bytes=config.get('receive_bytes',4*2**30),timeout=120))
        self.rdma_receiver = None
        if config.get('local_rdma_address'):
            from .rdma import RDMAServer
            self.rdma_receiver = stack.enter_context(Receiver(
                host=config['local_rdma_address'], server_class=RDMAServer,
                max_bytes=config.get('receive_bytes',4*2**30), timeout=120))
        self.egress_sender = None
        self.connections=config.get('connections',4)
        from .batch_decode import BatchDecoder
        self.batcher=BatchDecoder(self.backend,self.lock,max_batch=config.get('max_batch',4),wait_s=config.get('batch_wait_s',.005))
        stack.callback(self.batcher.close)

    def sync(self):
        if str(self.backend.device).startswith('cuda'): self.torch.cuda.synchronize()

    def handle(self,data):
        from .backend import restore_state, store_state, prefix_fingerprints, handoff_delta, merge_handoff
        from .transport import StreamSender, assemble_chunks
        op=data['operation']; backend=self.backend
        if op=='health':
            return dict(role=self.role,cache_url=self.node.url,namespace=self.pool.namespace,
                rdma_receiver=list(self.rdma_receiver.address) if self.rdma_receiver else None,
                receiver=list(self.receiver.address),vocab_size=backend.vocab_size,
                model_id=backend.model_id,revision=backend.revision,kernels=getattr(backend,'kernel_provenance',{}),pid=__import__('os').getpid())
        if op=='link':
            sender = self.egress_sender
            return sender.snapshot() if sender is not None else {}
        if op=='decode':
            with self.lock:
                if self.role!='decode': raise RuntimeError('worker is not a decoder')
                state=self.pending.pop(data['request'])
            return self.batcher.decode(state,data['output_tokens'])
        with self.lock:
            if op=='role':
                if self.pending: raise RuntimeError('worker has reserved decode requests')
                if data['role'] not in ('prefill','decode'): raise ValueError('invalid role')
                self.role=data['role']; return {'role':self.role}
            if op=='discard':
                self.pending.pop(data['request'],None)
                self.receiver.discard(data['request'])
                if self.rdma_receiver: self.rdma_receiver.discard(data['request'])
                return {}
            if op=='clear':
                if self.pending: raise RuntimeError('cannot reset active worker')
                for entry in self.pool.inventory(): self.pool.delete(entry['tokens'])
                return {}
            if op=='stats': return self.pool.stats()
            if op=='prepare':
                if self.role!='decode': raise RuntimeError('worker is not a decoder')
                if len(self.pending)>=8: raise RuntimeError('decode reservations full')
                tokens=data['tokens']; self._tokens(tokens)
                request=data['request']
                if request in self.pending: raise ValueError('duplicate request')
                state=restore_state(self.pool,backend,tokens)
                self.pending[request]=state
                length = 0 if state is None else int(state['length'])
                return {'prefix_tokens':length, 'fingerprints': prefix_fingerprints(backend,state,length)
                        if not data.get('cached_only') else {}}
            if op=='prefill':
                if self.role!='prefill': raise RuntimeError('worker is not a prefill worker')
                tokens=data['tokens']; self._tokens(tokens)
                state=restore_state(self.pool,backend,tokens)
                prefix=0 if state is None else int(state['length'])
                tensor=self.torch.tensor(tokens[prefix:],dtype=self.torch.int64,device=backend.device)
                self.sync(); start=time.perf_counter(); ready=[]
                connector = None
                if data.get('ssh_stream_ports'):
                    from .transport import ssh_stream_connector
                    connector = ssh_stream_connector(data['destination'], data['ssh_stream_ports'], self.connections)
                if data.get('rdma'):
                    from .rdma import connect
                    connector = connect
                with ExitStack() as sending, StreamSender(tuple(data['destination']),data['request'],min(self.connections,self.pool.capacity_blocks),pool=self.pool,
                        timeout=120,connect=connector,rate_limiter=self.rate_limiter if data.get('rate_limit') else None) as sender:
                    if data.get('rate_limit'):
                        self.egress_sender = sender
                        sending.callback(setattr, self, 'egress_sender', None)
                    def on_layer(i,layer):
                        ready.append(time.perf_counter())
                        sender.submit_blob(str(i),backend.encode(handoff_delta(backend,layer,data['destination_prefix'],data.get('fingerprints',{}))),
                            chunk_bytes=min(4*2**20,max(1,self.pool.capacity_blocks*self.pool.block_bytes//min(self.connections,self.pool.capacity_blocks))))
                    if tensor.numel(): logits,state=backend.prefill(tensor,state,on_layer=on_layer)
                    else:
                        logits,state=backend.forward(tensor,state)
                        for i in range(backend.layers):
                            on_layer(i,{k:v for k,v in state.items() if not k.startswith('layer.') or k.startswith(f'layer.{i}.')})
                    self.sync(); compute_end=time.perf_counter()
                stored=store_state(self.pool,backend,state)
                return dict(first_token=int(logits.reshape(-1,backend.vocab_size)[-1].argmax()),
                    prefix_tokens=prefix,prefill_s=compute_end-start,sender=sender.stats,stored=stored,
                    pipelined=bool(sender.stats['first_send_at'] and sender.stats['first_send_at']<ready[-1]),
                    layer_ready_s=[t-start for t in ready],compute_end_s=compute_end-start)
            if op=='consume_cached':
                state=self.pending[data['request']]
                if state is None or int(state['length'])!=data['length']: raise ValueError('incomplete cache hit')
                return {'first_token':int(state['last_logits'].reshape(-1,backend.vocab_size)[-1].argmax())}
            if op=='consume':
                if self.role!='decode': raise RuntimeError('worker is not a decoder')
                request=data['request']
                if request not in self.pending: raise ValueError('missing decode reservation')
                base=self.pending.pop(request)
                receiver = self.rdma_receiver if data.get('rdma') else self.receiver
                frames=assemble_chunks(receiver.take(request))
                if set(frames)!={str(i) for i in range(backend.layers)}: raise ValueError('incomplete layers')
                change={}
                for i in range(backend.layers): change.update(backend.decode(frames[str(i)]))
                state=merge_handoff(backend,base,change)
                self.sync(); logits=state['last_logits']
                if int(logits.reshape(-1,backend.vocab_size)[-1].argmax())!=data['first_token']:
                    raise ValueError('transferred logits do not match prefill token')
                store_state(self.pool,backend,state)
                self.pending[request]=state
                return {'first_token':data['first_token']}
            if op=='warmup':
                tokens=data['tokens']; self._tokens(tokens)
                vector=self.torch.tensor(tokens,dtype=self.torch.int64,device=backend.device)
                for batch in ([1,2,4] if backend.sequence_axis==2 else [1]):
                    inputs=vector if batch==1 else vector.unsqueeze(0).repeat(batch,1)
                    logits,state=backend.prefill(inputs)
                    token=logits.reshape(batch,backend.vocab_size).argmax(-1)
                    backend.forward(token.reshape(batch,1) if batch>1 else token.reshape(1),state)
                self.sync(); return {'warmed':True}
            if op=='reference':
                tokens=data['tokens']; self._tokens(tokens)
                tensor=self.torch.tensor(tokens,dtype=self.torch.int64,device=backend.device)
                logits,state=backend.prefill(tensor); generated=[]
                count(data['output_tokens'],'output tokens',zero=False)
                for i in range(data['output_tokens']):
                    token=logits.reshape(-1,backend.vocab_size)[-1].argmax().reshape(1); generated.append(int(token))
                    if i+1<data['output_tokens']: logits,state=backend.forward(token,state)
                return {'generated_tokens':generated}
            raise ValueError('unknown operation')

    def _tokens(self,tokens):
        if not isinstance(tokens,list) or not 0<len(tokens)<=262144 or any(type(t) is not int or not 0<=t<self.backend.vocab_size for t in tokens):
            raise ValueError('invalid token list')


def _serve(config,ready):
    try:
        with ExitStack() as stack:
            worker=ModelWorker(config,stack)
            class Handler(BaseHTTPRequestHandler):
                def log_message(self,*args): pass
                def do_POST(self):
                    try:
                        if not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+worker.token):
                            self.send_error(401); return
                        size=int(self.headers.get('Content-Length','0'))
                        if not 0<size<=8*2**20: raise ValueError('invalid request size')
                        data=json.loads(self.rfile.read(size))
                        if data.get('operation')=='stop':
                            threading.Thread(target=server.shutdown,daemon=True).start(); result={}
                        else: result=worker.handle(data)
                    except Exception as error:
                        result={'error':f'{type(error).__name__}: {error}'}
                        if getattr(error,'transport_stats',None): result['transport_stats']=error.transport_stats
                    raw=json.dumps(result).encode(); self.send_response(200)
                    self.send_header('Content-Length',str(len(raw))); self.end_headers(); self.wfile.write(raw)
            server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
            ready.put({'url':f'http://127.0.0.1:{server.server_port}',**worker.handle({'operation':'health'})})
            server.serve_forever(poll_interval=.05)
            server.server_close()
    except BaseException as error:
        ready.put({'error':f'{type(error).__name__}: {error}'}); raise


class WorkerProcess:
    def __init__(self,config):
        self.config=config; self.process=None

    def __enter__(self):
        ctx=multiprocessing.get_context('spawn'); ready=ctx.Queue()
        self.process=ctx.Process(target=_serve,args=(self.config,ready)); self.process.start()
        try:
            self.info=ready.get(timeout=600)
            if 'error' in self.info: raise RuntimeError(self.info['error'])
            return self
        except BaseException:
            self.__exit__(); raise
        finally: ready.close(); ready.join_thread()

    def call(self,operation,**data): return call(self.info['url'],self.config['token'],operation,**data)

    def __exit__(self,*exc):
        if self.process is not None:
            if hasattr(self,'info') and 'url' in self.info:
                try: self.call('stop')
                except Exception: pass
            else: self.process.terminate()
            self.process.join(timeout=30)
            if self.process.is_alive(): self.process.terminate(); self.process.join(timeout=5)
            if self.process.is_alive(): self.process.kill(); self.process.join()
