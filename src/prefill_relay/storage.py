"""Transactional SSD-backed hybrid checkpoints with a bounded RAM read cache."""
from collections import OrderedDict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
import threading
import time
from uuid import uuid4

from .analytics import count
from .cache import Match


class PersistentHybridCache:
    """One process owns a database; network clients share its serialized operations."""
    def __init__(self, path, block_tokens=16, block_bytes=4096, capacity_blocks=1024,
                 namespace='default', ram_bytes=16 * 1024 * 1024):
        for value, name in [(block_tokens, 'block_tokens'), (block_bytes, 'block_bytes'),
                            (capacity_blocks, 'capacity_blocks')]:
            count(value, name, zero=False)
        count(ram_bytes, 'ram_bytes')
        if not isinstance(namespace, str) or not 0 < len(namespace) <= 512:
            raise ValueError('namespace must contain 1..512 characters')
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.block_tokens, self.block_bytes = block_tokens, block_bytes
        self.capacity_blocks, self.ram_limit = capacity_blocks, ram_bytes
        self._lock = threading.RLock()
        self._ram, self._ram_size = OrderedDict(), 0
        self._hits = self._ssd_reads = self._corrupt = self._evictions = 0
        self._clock = 0
        self._closed = False
        self._owner = self.path.with_suffix(self.path.suffix + '.lock').open('a+b')
        try:
            fcntl.flock(self._owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._owner.close()
            raise RuntimeError('cache database already has an owner') from error
        try:
            self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
            self._db.execute('PRAGMA journal_mode=WAL')
            self._db.execute('PRAGMA synchronous=FULL')
            self._db.execute('PRAGMA foreign_keys=ON')
            self._db.executescript('''
                CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS blocks (
                    id TEXT PRIMARY KEY, data BLOB NOT NULL, checksum TEXT NOT NULL,
                    touched INTEGER NOT NULL, units INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY, tokens TEXT NOT NULL, full_keys TEXT NOT NULL,
                    recurrent_keys TEXT NOT NULL, touched INTEGER NOT NULL, length INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS refs (
                    checkpoint TEXT REFERENCES checkpoints(id) ON DELETE CASCADE,
                    block TEXT REFERENCES blocks(id) ON DELETE CASCADE,
                    PRIMARY KEY(checkpoint, block));
                CREATE INDEX IF NOT EXISTS refs_block ON refs(block);
                CREATE TABLE IF NOT EXISTS transfers (
                    handle TEXT PRIMARY KEY, data BLOB NOT NULL, checksum TEXT NOT NULL,
                    units INTEGER NOT NULL);
            ''')
            settings = dict(namespace=namespace, block_tokens=block_tokens,
                            block_bytes=block_bytes, capacity_blocks=capacity_blocks)
            with self._transaction():
                block_columns = {row[1] for row in self._db.execute('PRAGMA table_info(blocks)')}
                if 'units' not in block_columns:
                    self._db.execute('ALTER TABLE blocks ADD COLUMN units INTEGER NOT NULL DEFAULT 1')
                self._db.execute('CREATE INDEX IF NOT EXISTS blocks_units ON blocks(units)')
                columns = {row[1] for row in self._db.execute('PRAGMA table_info(checkpoints)')}
                if 'length' not in columns:
                    self._db.execute('ALTER TABLE checkpoints ADD COLUMN length INTEGER NOT NULL DEFAULT 0')
                    for identity, tokens in self._db.execute('SELECT id,tokens FROM checkpoints').fetchall():
                        self._db.execute('UPDATE checkpoints SET length=? WHERE id=?', (len(json.loads(tokens)), identity))
                prior = dict(self._db.execute('SELECT key,value FROM settings'))
                expected = {k: json.dumps(v) for k, v in settings.items()}
                if prior and prior != expected:
                    raise ValueError('cache namespace or storage geometry does not match database')
                self._db.executemany('INSERT OR IGNORE INTO settings VALUES (?,?)', expected.items())
                self._recovered = self._db.execute('SELECT COUNT(*) FROM transfers').fetchone()[0]
                # Restarted senders cannot confirm outstanding transfers; reusable data is unaffected.
                self._db.execute('DELETE FROM transfers')
                self._clock = self._db.execute('SELECT COALESCE(MAX(touched),0) FROM blocks').fetchone()[0]
        except BaseException:
            if hasattr(self, '_db'): self._db.close()
            self._owner.close()
            raise

    @contextmanager
    def _transaction(self):
        with self._lock:
            if self._closed: raise RuntimeError('cache is closed')
            self._db.execute('BEGIN IMMEDIATE')
            try:
                yield
            except BaseException:
                self._db.rollback()
                self._ram.clear(); self._ram_size = 0
                raise
            else:
                self._db.commit()

    def _tick(self):
        self._clock = max(time.time_ns(), self._clock + 1)
        return self._clock

    def _identities(self, tokens, lengths=None):
        if len(tokens) > 1048576: raise ValueError('too many tokens')
        digest = hashlib.sha256(self.namespace.encode())
        identities = []
        for index, token in enumerate(tokens, 1):
            if type(token) is not int or not 0 <= token < 2**63:
                raise ValueError('tokens must be nonnegative int64 values')
            digest.update(struct.pack('!Q', token))
            if (lengths is None and (index % self.block_tokens == 0 or index == len(tokens))) or (
                    lengths is not None and index in lengths):
                identities.append((index, digest.hexdigest()))
        return identities

    def _candidates(self, tokens):
        lengths = {row[0] for row in self._db.execute('SELECT DISTINCT length FROM checkpoints WHERE length<=?', (len(tokens),))}
        return reversed(self._identities(tokens, lengths))

    def _group(self, name):
        if not isinstance(name, str) or not 0 < len(name) <= 128:
            raise ValueError('invalid cache group name')

    @staticmethod
    def _key(*parts):
        return hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).hexdigest()

    def _remember(self, key, value):
        old = self._ram.pop(key, None)
        if old is not None: self._ram_size -= len(old)
        if len(value) > self.ram_limit: return
        while self._ram and self._ram_size + len(value) > self.ram_limit:
            _, old = self._ram.popitem(last=False); self._ram_size -= len(old)
        self._ram[key] = value; self._ram_size += len(value)

    def _forget(self, key):
        value = self._ram.pop(key, None)
        if value is not None: self._ram_size -= len(value)

    def _read(self, key):
        if key in self._ram:
            self._hits += 1; self._ram.move_to_end(key)
            return self._ram[key]
        row = self._db.execute('SELECT data,checksum FROM blocks WHERE id=?', (key,)).fetchone()
        if row is None: return None
        payload, checksum = row; self._ssd_reads += 1
        if hashlib.sha256(payload).hexdigest() != checksum:
            self._corrupt += 1; self._delete_block(key)
            return None
        self._remember(key, payload)
        return payload

    def _delete_block(self, key):
        self._db.execute('DELETE FROM checkpoints WHERE id IN (SELECT checkpoint FROM refs WHERE block=?)', (key,))
        self._db.execute('DELETE FROM blocks WHERE id=?', (key,)); self._forget(key)

    def _used(self):
        reusable = self._db.execute('SELECT COALESCE(SUM(units),0) FROM blocks').fetchone()[0]
        transfer = self._db.execute('SELECT COALESCE(SUM(units),0) FROM transfers').fetchone()[0]
        return reusable, transfer

    @property
    def used_blocks(self):
        with self._lock: return sum(self._used())

    def _reserve(self, units, protected=None):
        protected = {} if protected is None else protected
        reusable, transfer = self._used()
        if units + sum(protected.values()) + transfer > self.capacity_blocks:
            raise MemoryError('cache capacity exhausted by request or pinned transfers')
        needed = max(0, reusable + transfer + units - self.capacity_blocks)
        if not needed: return
        victims = [row for row in self._db.execute('SELECT id,units FROM blocks ORDER BY touched,id')
                   if row[0] not in protected]
        if sum(size for _, size in victims) < needed: raise MemoryError('insufficient evictable blocks')
        for key, size in victims:
            self._delete_block(key); self._evictions += 1; needed -= size
            if needed <= 0: break

    def put(self, tokens, full, recurrent):
        tokens = tuple(tokens); identities = [identity for _, identity in self._identities(tokens)]
        if not tokens: raise ValueError('prefix must be nonempty')
        if not full or not recurrent: raise ValueError('full and recurrent groups are required')
        incoming, full_keys, recurrent_keys = {}, {}, {}
        for group, values in full.items():
            self._group(group)
            if len(values) != len(identities): raise ValueError('wrong full-attention block count')
            keys = []
            for identity, value in zip(identities, values):
                if not isinstance(value, bytes) or not value:
                    raise ValueError('invalid full-attention block payload')
                if len(value)>self.block_bytes*self.capacity_blocks:
                    raise MemoryError('full-attention block exceeds cache capacity')
                key = self._key('full', group, identity, hashlib.sha256(value).hexdigest())
                keys.append(key); incoming[key] = value
            full_keys[group] = keys
        for group, payload in recurrent.items():
            self._group(group)
            if not isinstance(payload, bytes) or not payload: raise ValueError('invalid recurrent payload')
            key = self._key('recurrent', group, identities[-1], hashlib.sha256(payload).hexdigest())
            recurrent_keys[group] = [key]
            incoming[key] = payload
        with self._transaction():
            identity = identities[-1]
            full_json, recurrent_json = json.dumps(full_keys, sort_keys=True), json.dumps(recurrent_keys, sort_keys=True)
            prior = self._db.execute('SELECT full_keys,recurrent_keys FROM checkpoints WHERE id=?', (identity,)).fetchone()
            # Validate before reserving: eviction must not erase a conflicting checkpoint.
            if prior and prior != (full_json, recurrent_json): raise ValueError('checkpoint group schema conflict')
            existing = {}
            incoming_units = {key: math.ceil(len(value) / self.block_bytes) for key, value in incoming.items()}
            for key, value in incoming.items():
                row = self._db.execute('SELECT checksum,units FROM blocks WHERE id=?', (key,)).fetchone()
                if row:
                    if row[0] != hashlib.sha256(value).hexdigest():
                        raise ValueError('conflicting state for the same model and prefix')
                    existing[key] = row[1]
            self._reserve(sum(incoming_units.values()) - sum(existing.values()), protected=existing)
            now = self._tick()
            for key, value in incoming.items():
                self._db.execute('INSERT INTO blocks VALUES (?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data,touched=excluded.touched',
                                 (key, value, hashlib.sha256(value).hexdigest(), now, incoming_units[key]))
                self._remember(key, value)
            record = (identity, json.dumps(tokens), full_json, recurrent_json, now, len(tokens))
            self._db.execute('INSERT OR REPLACE INTO checkpoints VALUES (?,?,?,?,?,?)', record)
            self._db.executemany('INSERT INTO refs VALUES (?,?)', ((identity, key) for key in incoming))

    def match_length(self, tokens):
        """Metadata-only candidate length; match() verifies bytes before reuse."""
        tokens = tuple(tokens)
        with self._lock:
            for length, identity in self._candidates(tokens):
                if self._db.execute('SELECT 1 FROM checkpoints WHERE id=?', (identity,)).fetchone(): return length
        return 0

    def match(self, tokens):
        tokens = tuple(tokens)
        with self._transaction():
            for length, identity in self._candidates(tokens):
                row = self._db.execute('SELECT full_keys,recurrent_keys FROM checkpoints WHERE id=?', (identity,)).fetchone()
                if not row: continue
                full, recurrent = map(json.loads, row)
                keys = {key for groups in (full, recurrent) for values in groups.values() for key in values}
                values = {key: self._read(key) for key in keys}
                if any(value is None for value in values.values()): continue
                now = self._tick()
                self._db.executemany('UPDATE blocks SET touched=? WHERE id=?', ((now, key) for key in keys))
                self._db.execute('UPDATE checkpoints SET touched=? WHERE id=?', (now, identity))
                return Match(length,
                             {g: tuple(values[k] for k in ks) for g, ks in full.items()},
                             {g: b''.join(values[k] for k in ks) for g, ks in recurrent.items()})
        return Match(0, {}, {})

    def stage_transfer(self, payload):
        if not isinstance(payload, bytes) or not payload: raise ValueError('nonempty bytes required')
        with self._transaction():
            units = math.ceil(len(payload) / self.block_bytes); self._reserve(units)
            handle = uuid4().hex
            self._db.execute('INSERT INTO transfers VALUES (?,?,?,?)',
                             (handle, payload, hashlib.sha256(payload).hexdigest(), units))
            return handle

    def transfer_payload(self, handle):
        with self._lock:
            row = self._db.execute('SELECT data,checksum FROM transfers WHERE handle=?', (handle,)).fetchone()
            if row is None: raise KeyError(handle)
            if hashlib.sha256(row[0]).hexdigest() != row[1]: raise ValueError('transfer checksum mismatch')
            return row[0]

    def complete_transfer(self, handle, acknowledged):
        if acknowledged is not True: raise ValueError('receiver acknowledgement required')
        self.abort_transfer(handle)

    def abort_transfer(self, handle):
        with self._transaction():
            if not self._db.execute('DELETE FROM transfers WHERE handle=?', (handle,)).rowcount:
                raise KeyError(handle)

    def delete(self, tokens):
        tokens = tuple(tokens); ids = self._identities(tokens)
        if not ids: return False
        with self._transaction():
            changed = bool(self._db.execute('DELETE FROM checkpoints WHERE id=?', (ids[-1][1],)).rowcount)
            orphaned = [r[0] for r in self._db.execute('SELECT id FROM blocks WHERE id NOT IN (SELECT block FROM refs)')]
            for key in orphaned: self._db.execute('DELETE FROM blocks WHERE id=?', (key,)); self._forget(key)
            return changed

    def inventory(self):
        with self._lock:
            return [dict(tokens=json.loads(tokens), length=len(json.loads(tokens)), touched=touched)
                    for tokens, touched in self._db.execute('SELECT tokens,touched FROM checkpoints ORDER BY touched DESC')]

    def stats(self):
        with self._lock:
            reusable, transfer = self._used()
            return dict(namespace=self.namespace, block_tokens=self.block_tokens, block_bytes=self.block_bytes,
                        capacity_blocks=self.capacity_blocks, used_blocks=reusable + transfer,
                        prefix_blocks=reusable, transfer_blocks=transfer, ram_bytes=self._ram_size,
                        ram_limit=self.ram_limit, ram_hits=self._hits, ssd_reads=self._ssd_reads,
                        evictions=self._evictions, corrupt_blocks=self._corrupt, recovered_transfers=self._recovered,
                        transfers=self._db.execute('SELECT COUNT(*) FROM transfers').fetchone()[0],
                        checkpoints=self._db.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0],
                        ssd_payload_bytes=self._db.execute('SELECT COALESCE(SUM(LENGTH(data)),0) FROM blocks').fetchone()[0]
                        + self._db.execute('SELECT COALESCE(SUM(LENGTH(data)),0) FROM transfers').fetchone()[0])

    def close(self):
        with self._lock:
            if not self._closed:
                self._db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
                self._db.close(); self._owner.close(); self._closed = True
                self._ram.clear(); self._ram_size = 0

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()
