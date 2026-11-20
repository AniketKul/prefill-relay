"""Live cache-aware placement and budgeted replication across cache nodes."""
from dataclasses import dataclass
import math
import threading

from .analytics import count
from .cache_service import StalePrefix, apply_delta, checkpoint_manifest, prepare_delta, _json_bytes


@dataclass(frozen=True)
class Migration:
    source: tuple[str, str]
    target: tuple[str, str]
    length: int = 0
    bytes_transferred: int = 0
    status: str = 'no_match'


class GlobalDirectory:
    """Explicit membership with live probes; unreachable nodes are excluded.

    A directory restart requires re-registering nodes. Persistent data remains
    on the nodes. This is a placement service, not a consensus or ownership log.
    """
    def __init__(self):
        self._nodes, self._loads = {}, {}
        self._lock = threading.RLock()

    def register(self, cluster, node, client):
        for name in (cluster, node):
            if not isinstance(name, str) or not 0 < len(name) <= 256: raise ValueError('invalid node identity')
        key = cluster, node, client.namespace
        with self._lock:
            if key in self._nodes and self._nodes[key] is not client:
                raise ValueError('node and namespace already registered')
            self._nodes[key] = client
            self._loads.setdefault((cluster, node), 0.0)

    def unregister(self, cluster, node, namespace):
        with self._lock: return self._nodes.pop((cluster, node, namespace), None) is not None

    def update_load(self, cluster, node, load):
        if isinstance(load, bool) or not isinstance(load, (int, float)) or not math.isfinite(load) or load < 0:
            raise ValueError('load must be finite and nonnegative')
        with self._lock: self._loads[(cluster, node)] = float(load)

    def _candidates(self, namespace, cluster=None):
        with self._lock:
            return [(c, n, client, self._loads[(c, n)]) for (c, n, ns), client in self._nodes.items()
                    if ns == namespace and (cluster is None or c == cluster)]

    def best(self, cluster, namespace, tokens, nodes=None):
        """Return (node, reusable tokens), preferring lower load on ties."""
        ranked = []
        for c, node, client, load in self._candidates(namespace, cluster):
            if nodes is not None and node not in nodes:
                continue
            try: length = client.match_length(tokens)
            except (ConnectionError, TimeoutError): continue
            ranked.append((length, -load, node))
        if not ranked: return None, 0
        length, _, node = max(ranked)
        return node, length

    def fetch(self, cluster, node, namespace, tokens):
        with self._lock: client = self._nodes[(cluster, node, namespace)]
        return client.match(tokens)

    def _client(self, address, namespace):
        if not isinstance(address, tuple) or len(address) != 2: raise ValueError('address must be (cluster, node)')
        with self._lock: return self._nodes[(*address, namespace)]

    def migrate(self, source, target, namespace, tokens, budget_bytes):
        """Replicate a verified checkpoint, charging serialized state bytes.

        The caller supplies the available link budget for this operation. Wire
        framing and control requests are outside that payload budget. Transfer
        staging shares source cache capacity and can evict reusable entries.
        """
        count(budget_bytes, 'budget_bytes')
        if source == target: return Migration(source, target, status='already_local')
        sender, receiver = self._client(source, namespace), self._client(target, namespace)
        if (sender.block_tokens, sender.block_bytes) != (receiver.block_tokens, receiver.block_bytes):
            raise ValueError('cache node block geometry must match')
        tokens = tuple(tokens)
        length = sender.match_length(tokens)
        if not length: return Migration(source, target)
        def manifest(node):
            return node.known(tokens[:length]) if hasattr(node, 'known') else checkpoint_manifest(node.match(tokens[:length]))
        target_manifest = manifest(receiver)
        known = target_manifest['hashes']
        if target_manifest['length'] == length:
            source_manifest = manifest(sender)
            if source_manifest['length'] != length:
                return Migration(source, target, status='stale_prefix')
            if 'recurrent_hashes' in target_manifest and source_manifest == target_manifest:
                return Migration(source, target, length, status='already_cached')
            return Migration(source, target, status='different_checkpoint')
        if hasattr(sender, 'prepare_delta'):
            prepared = sender.prepare_delta(tokens, known, budget_bytes, rate_limit=source[0] != target[0])
        else:
            prepared = prepare_delta(sender, tokens, known, budget_bytes)
        if prepared['status'] != 'ready':
            return Migration(source, target, prepared['length'], status=prepared['status'])
        handle = prepared['handle']
        acknowledged = False
        try:
            payload_bytes = len(_json_bytes(prepared['delta']))
            if payload_bytes != prepared['payload_bytes'] or payload_bytes > budget_bytes:
                raise ValueError('delta payload size does not match its budget')
            length = prepared['length']
            if not 0 < length <= len(tokens) or prepared['delta']['length'] != length:
                raise ValueError('delta checkpoint length mismatch')
            if hasattr(receiver, 'apply_delta'):
                if receiver.apply_delta(tokens[:length], prepared['delta'], known) != length:
                    raise RuntimeError('receiver did not verify the checkpoint')
            else:
                apply_delta(receiver, tokens[:length], prepared['delta'], known)
            sender.complete_transfer(handle, acknowledged=True)
            acknowledged = True
        except StalePrefix:
            return Migration(source, target, 0, payload_bytes, 'stale_prefix')
        finally:
            if not acknowledged:
                try: sender.abort_transfer(handle)
                except (KeyError, ConnectionError): pass
        return Migration(source, target, length, payload_bytes, 'transferred')

    def rebalance(self, namespace, tokens, loads=None, budget_bytes=0, allow_cross_cluster=False):
        """Copy a hot prefix from the busiest holder to an emptier eligible node."""
        if loads is not None:
            for (cluster, node), load in loads.items(): self.update_load(cluster, node, load)
        count(budget_bytes, 'budget_bytes')
        candidates = []
        for cluster, node, client, load in self._candidates(namespace):
            try: length = client.match_length(tokens)
            except (ConnectionError, TimeoutError): continue
            candidates.append(((cluster, node), length, load))
        sources = sorted((x for x in candidates if x[1]), key=lambda x: (x[2], x[1]), reverse=True)
        for source, length, source_load in sources:
            targets = [(target, load) for target, target_length, load in candidates
                       if target != source and load < source_load and target_length < length
                       and (allow_cross_cluster or target[0] == source[0])]
            targets.sort(key=lambda x: (x[0][0] != source[0], x[1], x[0]))
            if targets:
                return self.migrate(source, targets[0][0], namespace, tokens, budget_bytes)
        return Migration(('', ''), ('', ''), status='balanced')
