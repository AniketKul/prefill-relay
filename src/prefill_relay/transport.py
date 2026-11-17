"""Bounded layer-wise multi-connection TCP transport with application ACKs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import socket
import socketserver
import struct
import threading
import time
import sys
import math
from collections import OrderedDict
from .analytics import count, positive

MAX_HEADER=4096
MAX_FRAME=64*1024*1024


def _read(sock,n):
    chunks=[]
    while n:
        data=sock.recv(min(n,65536))
        if not data: raise ConnectionError('truncated transfer')
        chunks.append(data); n-=len(data)
    return b''.join(chunks)


def _name(value):
    if not isinstance(value,str) or not value or len(value)>128:
        raise ValueError('transfer and layer identifiers must have 1-128 characters')


def tcp_metrics(sock):
    if not sys.platform.startswith('linux') or not hasattr(socket,'TCP_INFO'):
        return {'tcp_info_available':False,'rtt_ms':0.,'retransmissions':0}
    try:
        info=sock.getsockopt(socket.IPPROTO_TCP,socket.TCP_INFO,104)
        return {'tcp_info_available':True,'rtt_ms':struct.unpack_from('=I',info,68)[0]/1000,
                'retransmissions':struct.unpack_from('=I',info,100)[0]}
    except OSError:
        return {'tcp_info_available':False,'rtt_ms':0.,'retransmissions':0}


def assemble_chunks(frames):
    groups={}; complete={}
    for name,payload in frames.items():
        if ':' not in name:
            if name in groups: raise ValueError('mixed chunked and complete layer')
            complete[name]=payload; continue
        layer,index,total=name.rsplit(':',2)
        try: index,total=int(index),int(total)
        except ValueError as e: raise ValueError('invalid chunk name') from e
        if not 0<=index<total<=100000 or layer in complete: raise ValueError('invalid chunks')
        expected,chunks=groups.setdefault(layer,(total,{}))
        if expected!=total or index in chunks: raise ValueError('conflicting chunks')
        chunks[index]=payload
    for layer,(total,chunks) in groups.items():
        if len(chunks)!=total: raise ValueError('incomplete layer transfer')
        complete[layer]=b''.join(chunks[i] for i in range(total))
    return complete


def send_frame(address,request,layer,payload,timeout=10):
    _name(request); _name(layer); positive(timeout,'timeout')
    if not isinstance(payload,bytes) or not 0<len(payload)<=MAX_FRAME: raise ValueError('invalid payload size')
    header=json.dumps(dict(request=request,layer=layer,size=len(payload),sha256=hashlib.sha256(payload).hexdigest()),separators=(',',':')).encode()
    if len(header)>MAX_HEADER: raise ValueError('header too large')
    with socket.create_connection(address,timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(struct.pack('!I',len(header))+header)
        started=time.perf_counter()
        sock.sendall(payload)
        ack=_read(sock,3)
        if ack!=b'ACK': raise RuntimeError('receiver rejected transfer')
        return {**tcp_metrics(sock),'send_started_at':started}


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads=False
    allow_reuse_address=True


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        owner=self.server.owner; sock=self.request
        sock.settimeout(owner.timeout)
        reserved=0; registered=None
        try:
            n=struct.unpack('!I',_read(sock,4))[0]
            if not 0<n<=MAX_HEADER: raise ValueError('invalid header length')
            header=json.loads(_read(sock,n))
            if not isinstance(header,dict) or set(header)!={'request','layer','size','sha256'}: raise ValueError('invalid header')
            request,layer,size=header['request'],header['layer'],header['size']
            _name(request); _name(layer); count(size,'frame size',zero=False)
            if size>MAX_FRAME or not isinstance(header['sha256'],str) or len(header['sha256'])!=64: raise ValueError('invalid frame')
            with owner.lock:
                owner._prune_aborted()
                if request in owner.aborted: raise ValueError('request aborted')
                owner.active[request]=owner.active.get(request,0)+1; registered=request
                prior=owner.frames.get(request,{}).get(layer)
                if prior is not None:
                    if len(prior)!=size or hashlib.sha256(prior).hexdigest()!=header['sha256']: raise ValueError('conflicting retry')
                    # Retry is checked without allocating another full payload.
                else:
                    if owner.buffered_bytes+owner.pending_bytes+size>owner.max_bytes: raise MemoryError('receiver full')
                    owner.pending_bytes+=size; reserved=size
            digest=hashlib.sha256(); pieces=[]; remaining=size
            while remaining:
                chunk=_read(sock,min(remaining,65536)); digest.update(chunk); remaining-=len(chunk)
                if prior is None: pieces.append(chunk)
            if digest.hexdigest()!=header['sha256']: raise ValueError('checksum mismatch')
            if prior is None:
                payload=b''.join(pieces)
                with owner.lock:
                    if request in owner.aborted: raise ValueError('request aborted')
                    existing=owner.frames.setdefault(request,{}).get(layer)
                    if existing is not None and existing!=payload: raise ValueError('conflicting concurrent retry')
                    if existing is None:
                        owner.frames[request][layer]=payload; owner.buffered_bytes+=size
                    owner.pending_bytes-=reserved; reserved=0
            sock.sendall(b'ACK')
        except (OSError,ValueError,TypeError,KeyError,MemoryError):
            try: sock.sendall(b'ERR')
            except OSError: pass
        finally:
            with owner.lock:
                if reserved: owner.pending_bytes-=reserved
                if registered:
                    owner.active[registered]-=1
                    if not owner.active[registered]: del owner.active[registered]


class Receiver:
    """Reference trusted-network receiver. Default binding is loopback only."""
    def __init__(self,host='127.0.0.1',port=0,max_bytes=128*1024*1024,timeout=10):
        count(max_bytes,'max bytes',zero=False); positive(timeout,'timeout')
        self.max_bytes=max_bytes; self.timeout=timeout; self.frames={}
        self.buffered_bytes=0; self.pending_bytes=0; self.lock=threading.RLock()
        self.aborted=OrderedDict(); self.active={}
        self.server=_Server((host,port),_Handler); self.server.owner=self
        self.address=self.server.server_address
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.05},daemon=True)

    def __enter__(self): self.thread.start(); return self

    def take(self,request):
        """Consume only after the sender has acknowledged every expected layer."""
        with self.lock:
            frames=self.frames.pop(request)
            self.buffered_bytes-=sum(map(len,frames.values()))
            return frames

    def _prune_aborted(self):
        now=time.monotonic()
        for request,expiry in list(self.aborted.items()):
            if expiry<now and request not in self.active: del self.aborted[request]

    def discard(self,request):
        """Abort buffered request state after all its sender futures have settled."""
        _name(request)
        with self.lock:
            self._prune_aborted()
            if request not in self.aborted and len(self.aborted)>=10000: raise RuntimeError('abort history full; wait for receiver timeout')
            self.aborted[request]=time.monotonic()+2*self.timeout
            frames=self.frames.pop(request,{})
            self.buffered_bytes-=sum(map(len,frames.values()))

    def __exit__(self,*exc):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


class StreamSender:
    """Submit layers as computed, bounded to one outstanding frame per connection."""
    def __init__(self,address,request,connections=4,pool=None,timeout=10,retries=2,bandwidth_bytes_s=None,rate_limiter=None):
        count(connections,'connections',zero=False); positive(timeout,'timeout'); _name(request)
        count(retries,'retries'); self.retries=retries
        if bandwidth_bytes_s is not None: positive(bandwidth_bytes_s,'bandwidth')
        self.rate_limiter=rate_limiter; self.bandwidth=bandwidth_bytes_s; self.next_send=time.monotonic(); self.rate_lock=threading.Lock()
        self.address=address; self.request=request; self.pool=pool; self.timeout=timeout
        self.executor=ThreadPoolExecutor(max_workers=connections)
        self.slots=threading.Semaphore(connections); self.lock=threading.Lock(); self.futures=[]
        self.stats={'bytes':0,'failures':0,'queue_depth':0,'elapsed_s':0.,'retries':0,'retransmissions':0,'rtt_ms':0.,'tcp_info_available':False,'first_send_at':None,'max_queue_depth':0}
        self.start=time.perf_counter()

    def __enter__(self): return self

    def submit(self,layer,payload):
        self.slots.acquire()
        handle=None
        try:
            if self.pool is not None: handle=self.pool.stage_transfer(payload)
            with self.lock:
                self.stats['queue_depth']+=1
                self.stats['max_queue_depth']=max(self.stats['max_queue_depth'],self.stats['queue_depth'])
            self.futures.append(self.executor.submit(self._send,layer,payload,handle))
        except BaseException:
            if handle is not None: self.pool.abort_transfer(handle)
            self.slots.release()
            raise

    def submit_blob(self,layer,payload,chunk_bytes=4*1024*1024):
        if ':' in layer: raise ValueError('blob layer names cannot contain colon')
        count(chunk_bytes,'chunk bytes',zero=False)
        if chunk_bytes>MAX_FRAME: raise ValueError('chunk exceeds frame limit')
        total=math.ceil(len(payload)/chunk_bytes)
        if not total: raise ValueError('empty blob')
        for index in range(total): self.submit(f'{layer}:{index}:{total}',payload[index*chunk_bytes:(index+1)*chunk_bytes])

    def _send(self,layer,payload,handle):
        try:
            if self.bandwidth:
                with self.rate_lock:
                    now=time.monotonic(); self.next_send=max(now,self.next_send)+len(payload)/self.bandwidth
                    delay=self.next_send-now
                time.sleep(delay)
            for attempt in range(self.retries+1):
                try:
                    if self.rate_limiter: self.rate_limiter(len(payload))
                    metrics=send_frame(self.address,self.request,layer,payload,self.timeout)
                    break
                except (OSError,ConnectionError):
                    if attempt==self.retries: raise
                    with self.lock: self.stats['retries']+=1
                    time.sleep(.01*2**attempt)
            with self.lock:
                when=metrics['send_started_at']
                self.stats['first_send_at']=min(self.stats['first_send_at'] or when,when)
                self.stats['retransmissions']+=metrics['retransmissions']
                self.stats['rtt_ms']=max(self.stats['rtt_ms'],metrics['rtt_ms'])
                self.stats['tcp_info_available']|=metrics['tcp_info_available']
            if handle is not None: self.pool.complete_transfer(handle,acknowledged=True)
            with self.lock: self.stats['bytes']+=len(payload)
        except BaseException:
            if handle is not None: self.pool.abort_transfer(handle)
            with self.lock: self.stats['failures']+=1
            raise
        finally:
            with self.lock: self.stats['queue_depth']-=1
            self.slots.release()

    def __exit__(self,exc_type,*rest):
        self.executor.shutdown(wait=True,cancel_futures=False)
        self.stats['elapsed_s']=time.perf_counter()-self.start
        if exc_type is None:
            for future in self.futures:
                try: future.result()
                except Exception as error:
                    error.transport_stats=dict(self.stats)
                    raise
