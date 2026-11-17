"""Bounded hybrid cache pool; opaque bytes permit any inference backend."""
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from uuid import uuid4
import json
import math
from .analytics import count


@dataclass(frozen=True)
class Match:
    length: int
    full: dict[str, tuple[bytes, ...]]
    recurrent: dict[str, bytes]


class HybridCache:
    def __init__(self,block_tokens=16,block_bytes=4096,capacity_blocks=1024,namespace='default'):
        for n,label in [(block_tokens,'block_tokens'),(block_bytes,'block_bytes'),(capacity_blocks,'capacity_blocks')]: count(n,label,zero=False)
        if not isinstance(namespace,str) or not namespace: raise ValueError('namespace required')
        self.block_tokens=block_tokens; self.block_bytes=block_bytes; self.capacity_blocks=capacity_blocks
        self.namespace=namespace; self._blocks=OrderedDict(); self._checkpoints={}; self._transfers={}
        self._lock=RLock()

    def _identity(self,tokens):
        for t in tokens: count(t,'token')
        return sha256(json.dumps([self.namespace,list(tokens)],separators=(',',':')).encode()).hexdigest()

    @property
    def used_blocks(self):
        with self._lock:
            return len(self._blocks)+sum(len(v) for v in self._transfers.values())

    def _reserve(self,n,protected=()):
        if n>self.capacity_blocks-sum(len(v) for v in self._transfers.values()):
            raise MemoryError('transfer-pinned pool capacity exhausted')
        victims=[key for key in self._blocks if key not in protected]
        needed=max(0,self.used_blocks+n-self.capacity_blocks)
        if len(victims)<needed: raise MemoryError('insufficient unpinned blocks')
        for key in victims[:needed]: del self._blocks[key]
        # A checkpoint with any evicted group cannot be advertised as reusable.
        for key,record in list(self._checkpoints.items()):
            if any(k not in self._blocks for k in record[2]): del self._checkpoints[key]

    def put(self,tokens,full,recurrent):
        tokens=tuple(tokens); self._identity(tokens)
        if not tokens or len(tokens)%self.block_tokens: raise ValueError('prefix must end on a complete block')
        if not full or not recurrent: raise ValueError('both full and recurrent groups required')
        chunks=len(tokens)//self.block_tokens; incoming={}; full_keys={}; recurrent_keys={}
        for group,blocks in full.items():
            if not isinstance(group,str) or not group or len(blocks)!=chunks: raise ValueError('invalid full group')
            keys=[]
            for i,payload in enumerate(blocks):
                if not isinstance(payload,bytes) or not 0<len(payload)<=self.block_bytes: raise ValueError('full block too large or empty')
                key=('full',group,self._identity(tokens[:(i+1)*self.block_tokens]))
                incoming[key]=payload; keys.append(key)
            full_keys[group]=tuple(keys)
        identity=self._identity(tokens)
        for group,payload in recurrent.items():
            if not isinstance(group,str) or not group or not isinstance(payload,bytes) or not payload: raise ValueError('invalid recurrent group')
            keys=[]
            for i in range(0,len(payload),self.block_bytes):
                key=('recurrent',group,identity,i//self.block_bytes)
                incoming[key]=payload[i:i+self.block_bytes]; keys.append(key)
            recurrent_keys[group]=tuple(keys)
        with self._lock:
            for key,payload in incoming.items():
                if key in self._blocks and self._blocks[key]!=payload: raise ValueError('conflicting state under same model/token identity')
            self._reserve(sum(k not in self._blocks for k in incoming),protected=incoming)
            self._blocks.update(incoming)
            for key in incoming: self._blocks.move_to_end(key)
            self._checkpoints[identity]=(full_keys,recurrent_keys,tuple(incoming),len(tokens))

    def match(self,tokens):
        tokens=tuple(tokens); self._identity(tokens)
        with self._lock:
            for length in range(len(tokens)//self.block_tokens*self.block_tokens,0,-self.block_tokens):
                record=self._checkpoints.get(self._identity(tokens[:length]))
                if record is None: continue
                full,recurrent,keys,_=record
                if any(k not in self._blocks for k in keys): continue
                for k in keys: self._blocks.move_to_end(k)
                return Match(length,{g:tuple(self._blocks[k] for k in ks) for g,ks in full.items()},{g:b''.join(self._blocks[k] for k in ks) for g,ks in recurrent.items()})
            return Match(0,{}, {})

    def stage_transfer(self,payload):
        if not isinstance(payload,bytes) or not payload: raise ValueError('nonempty bytes required')
        with self._lock:
            self._reserve(math.ceil(len(payload)/self.block_bytes))
            handle=uuid4().hex
            self._transfers[handle]=tuple(payload[i:i+self.block_bytes] for i in range(0,len(payload),self.block_bytes))
            return handle

    def complete_transfer(self,handle,acknowledged):
        if acknowledged is not True: raise ValueError('receiver acknowledgement required')
        self.abort_transfer(handle)

    def abort_transfer(self,handle):
        """Explicitly abandon an unsuccessful transfer, preserving reusable prefixes."""
        with self._lock: del self._transfers[handle]


class CacheDirectory:
    """In-process metadata view; queries the owner to avoid advertising evicted blocks."""
    def __init__(self): self.nodes={}

    def register(self,cluster,node,cache):
        if not cluster or not node: raise ValueError('cluster and node required')
        self.nodes[(cluster,node)]=cache

    def best(self,cluster,namespace,tokens):
        matches=[(node,c.match(tokens).length) for (cl,node),c in self.nodes.items() if cl==cluster and c.namespace==namespace]
        return max(matches,key=lambda v:v[1]) if matches else (None,0)
