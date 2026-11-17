"""Bounded layer-wise multi-connection TCP transport with application ACKs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import socket
import socketserver
import struct
import threading
import time
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


def send_frame(address,request,layer,payload,timeout=10):
    _name(request); _name(layer); positive(timeout,'timeout')
    if not isinstance(payload,bytes) or not 0<len(payload)<=MAX_FRAME: raise ValueError('invalid payload size')
    header=json.dumps(dict(request=request,layer=layer,size=len(payload),sha256=hashlib.sha256(payload).hexdigest()),separators=(',',':')).encode()
    if len(header)>MAX_HEADER: raise ValueError('header too large')
    with socket.create_connection(address,timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(struct.pack('!I',len(header))+header)
        sock.sendall(payload)
        ack=_read(sock,3)
        if ack!=b'ACK': raise RuntimeError('receiver rejected transfer')


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads=False
    allow_reuse_address=True


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        owner=self.server.owner; sock=self.request
        sock.settimeout(owner.timeout)
        reserved=0
        try:
            n=struct.unpack('!I',_read(sock,4))[0]
            if not 0<n<=MAX_HEADER: raise ValueError('invalid header length')
            header=json.loads(_read(sock,n))
            if not isinstance(header,dict) or set(header)!={'request','layer','size','sha256'}: raise ValueError('invalid header')
            request,layer,size=header['request'],header['layer'],header['size']
            _name(request); _name(layer); count(size,'frame size',zero=False)
            if size>MAX_FRAME or not isinstance(header['sha256'],str) or len(header['sha256'])!=64: raise ValueError('invalid frame')
            with owner.lock:
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
            if reserved:
                with owner.lock: owner.pending_bytes-=reserved


class Receiver:
    """Reference trusted-network receiver. Default binding is loopback only."""
    def __init__(self,host='127.0.0.1',port=0,max_bytes=128*1024*1024,timeout=10):
        count(max_bytes,'max bytes',zero=False); positive(timeout,'timeout')
        self.max_bytes=max_bytes; self.timeout=timeout; self.frames={}
        self.buffered_bytes=0; self.pending_bytes=0; self.lock=threading.RLock()
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

    def discard(self,request):
        """Abort buffered request state after all its sender futures have settled."""
        _name(request)
        with self.lock:
            frames=self.frames.pop(request,{})
            self.buffered_bytes-=sum(map(len,frames.values()))

    def __exit__(self,*exc):
        self.server.shutdown(); self.server.server_close(); self.thread.join()


class StreamSender:
    """Submit layers as computed, bounded to one outstanding frame per connection."""
    def __init__(self,address,request,connections=4,pool=None,timeout=10):
        count(connections,'connections',zero=False); positive(timeout,'timeout'); _name(request)
        self.address=address; self.request=request; self.pool=pool; self.timeout=timeout
        self.executor=ThreadPoolExecutor(max_workers=connections)
        self.slots=threading.Semaphore(connections); self.lock=threading.Lock(); self.futures=[]
        self.stats={'bytes':0,'failures':0,'queue_depth':0,'elapsed_s':0.}
        self.start=time.perf_counter()

    def __enter__(self): return self

    def submit(self,layer,payload):
        self.slots.acquire()
        handle=None
        try:
            if self.pool is not None: handle=self.pool.stage_transfer(payload)
            with self.lock: self.stats['queue_depth']+=1
            self.futures.append(self.executor.submit(self._send,layer,payload,handle))
        except BaseException:
            if handle is not None: self.pool.abort_transfer(handle)
            self.slots.release()
            raise

    def _send(self,layer,payload,handle):
        try:
            send_frame(self.address,self.request,layer,payload,self.timeout)
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
            for future in self.futures: future.result()
