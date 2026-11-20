"""Bounded authenticated HTTP access to a single persistent cache owner."""
import base64
import binascii
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .analytics import count
from .cache import Match


def encode_match(match):
    return dict(length=match.length,
                full={g: [base64.b64encode(v).decode('ascii') for v in values]
                      for g, values in match.full.items()},
                recurrent={g: base64.b64encode(v).decode('ascii') for g, v in match.recurrent.items()})


def _decode(value):
    if not isinstance(value, str): raise ValueError('payload must be base64 text')
    try: return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as error: raise ValueError('invalid base64 payload') from error


def decode_match(value):
    if not isinstance(value, dict): raise ValueError('state must be an object')
    full, recurrent = value.get('full'), value.get('recurrent')
    if not isinstance(full, dict) or not isinstance(recurrent, dict):
        raise ValueError('state groups must be objects')
    if len(full) > 1024 or len(recurrent) > 1024: raise ValueError('too many state groups')
    if any(not isinstance(v, list) for v in full.values()): raise ValueError('full blocks must be arrays')
    length = value.get('length', 0); count(length, 'length')
    return Match(length, {g: tuple(_decode(v) for v in vs) for g, vs in full.items()},
                 {g: _decode(v) for g, v in recurrent.items()})


def _json_bytes(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode('utf-8')


def prepare_delta(cache, tokens, known_blocks, budget_bytes, rate_limiter=None, max_response_bytes=None):
    """Build and pin missing blocks at their owner, before crossing a link."""
    count(budget_bytes, 'budget_bytes')
    if not isinstance(known_blocks, dict) or len(known_blocks) > 1024:
        raise ValueError('known blocks must be a bounded group mapping')
    for group, hashes in known_blocks.items():
        if not isinstance(group, str) or not 0 < len(group) <= 128 or not isinstance(hashes, list):
            raise ValueError('invalid known block group')
        if len(hashes) > 1048576: raise ValueError('too many known blocks')
        if any(not isinstance(value, str) or len(value) != 64 or
               any(c not in '0123456789abcdef' for c in value) for value in hashes):
            raise ValueError('known block hashes must be lowercase SHA256 digests')
    match = cache.match(tokens)
    if not match.length:
        return dict(status='no_match', length=0, payload_bytes=0, handle=None, delta=None)
    delta = encode_match(match)
    for group, values in match.full.items():
        known = known_blocks.get(group, [])
        for index, value in enumerate(values):
            if index < len(known) and hashlib.sha256(value).hexdigest() == known[index]:
                delta['full'][group][index] = None
    payload = _json_bytes(delta)
    if len(payload) > budget_bytes:
        return dict(status='budget_exceeded', length=match.length, payload_bytes=0, handle=None, delta=None)
    response = dict(status='ready', length=match.length, payload_bytes=len(payload), handle='0' * 32, delta=delta)
    if max_response_bytes is not None and len(_json_bytes(response)) > max_response_bytes:
        raise ValueError('delta response exceeds configured byte limit')
    handle = cache.stage_transfer(payload)
    try:
        if cache.transfer_payload(handle) != payload: raise ValueError('transfer integrity check failed')
        if rate_limiter is not None: rate_limiter(len(payload))
        response['handle'] = handle
        return response
    except BaseException:
        cache.abort_transfer(handle)
        raise


def block_hashes(match):
    return {group: [hashlib.sha256(block).hexdigest() for block in blocks]
            for group, blocks in match.full.items()}


def restore_delta(delta, cached, expected=None):
    """Reconstruct a complete checkpoint from a delta and retained local bytes."""
    if not isinstance(delta, dict) or not isinstance(delta.get('full'), dict):
        raise ValueError('invalid delta manifest')
    full = {}
    for group, encoded in delta['full'].items():
        if not isinstance(encoded, list): raise ValueError('invalid delta blocks')
        known, values = cached.full.get(group, ()), []
        for index, value in enumerate(encoded):
            if value is None:
                if index >= len(known): raise ValueError('delta references an unavailable local block')
                if expected is not None:
                    hashes = expected.get(group, [])
                    if index >= len(hashes) or hashlib.sha256(known[index]).hexdigest() != hashes[index]:
                        raise ValueError('cached block changed during migration')
                value = base64.b64encode(known[index]).decode('ascii')
            values.append(value)
        full[group] = values
    return decode_match(dict(length=delta.get('length'), full=full, recurrent=delta.get('recurrent')))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            request.close()
            return
        try: super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try: super().process_request_thread(request, client_address)
        finally: self.slots.release()


class CacheNode:
    """Serve an existing cache; only the owner chooses the on-disk database path.

    The cache outlives this server context. Remote deployments should provide TLS
    through a private authenticated proxy; this server implements plain HTTP.
    """
    def __init__(self, cache, host='127.0.0.1', port=0, token=None,
                 max_body_bytes=64 * 1024 * 1024, max_clients=32, timeout=15, rate_limiter=None):
        count(max_body_bytes, 'max_body_bytes', zero=False)
        count(max_clients, 'max_clients', zero=False)
        count(port, 'port')
        if port > 65535: raise ValueError('invalid port')
        if token is not None and (not isinstance(token, str) or not token): raise ValueError('invalid token')
        try: loopback = ipaddress.ip_address(host).is_loopback
        except ValueError: loopback = host == 'localhost'
        if not loopback and not token: raise ValueError('non-loopback listeners require an authentication token')
        if timeout <= 0: raise ValueError('timeout must be positive')
        self.cache, self.token, self.max_body_bytes = cache, token, max_body_bytes
        self.timeout = timeout
        if rate_limiter is not None and not callable(rate_limiter): raise ValueError('rate limiter must be callable')
        self.rate_limiter = rate_limiter
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass

            def setup(self):
                super().setup(); self.connection.settimeout(owner.timeout)

            def reply(self, status, value):
                body = _json_bytes(value)
                if len(body) > owner.max_body_bytes:
                    status, body = 413, b'{"error":"ValueError","message":"response exceeds configured byte limit"}'
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers(); self.wfile.write(body)

            def do_POST(self):
                result = None
                if owner.token is not None and not hmac.compare_digest(
                        self.headers.get('Authorization', '').encode(), ('Bearer ' + owner.token).encode()):
                    self.reply(401, dict(error='PermissionError', message='authentication required')); return
                try:
                    if self.headers.get('Transfer-Encoding'): raise ValueError('chunked requests are unsupported')
                    lengths = self.headers.get_all('Content-Length', [])
                    if len(lengths) != 1: raise ValueError('one Content-Length is required')
                    size = int(lengths[0])
                    if not 0 < size <= owner.max_body_bytes: raise ValueError('request exceeds configured byte limit')
                    raw = self.rfile.read(size)
                    if len(raw) != size: raise ValueError('incomplete request')
                    data = json.loads(raw)
                    if not isinstance(data, dict): raise ValueError('request must be an object')
                    if data.get('namespace') != owner.cache.namespace: raise ValueError('model namespace mismatch')
                    result = owner._dispatch(self.path, data)
                    self.reply(200, result)
                except (ValueError, TypeError, KeyError, MemoryError, RuntimeError) as error:
                    status = 507 if isinstance(error, MemoryError) else 400
                    self.reply(status, dict(error=type(error).__name__, message=str(error)))
                except (BrokenPipeError, ConnectionResetError, TimeoutError):
                    if self.path == '/prepare_delta' and result and result.get('handle'):
                        try: owner.cache.abort_transfer(result['handle'])
                        except KeyError: pass
                    self.close_connection = True

        self._server = _Server((host, port), Handler)
        self._server.slots = threading.BoundedSemaphore(max_clients)
        self._thread = None
        address, actual_port = self._server.server_address[:2]
        self.url = f'http://{address}:{actual_port}'

    def _dispatch(self, operation, data):
        cache = self.cache
        if operation == '/info': return cache.stats()
        if operation == '/inventory': return cache.inventory()
        if operation in ('/match', '/match_length', '/put', '/delete', '/prepare_delta', '/known', '/apply_delta'):
            tokens = data.get('tokens')
            if not isinstance(tokens, list) or len(tokens) > 1048576: raise ValueError('invalid tokens')
            if operation == '/known':
                match = cache.match(tokens)
                return dict(length=match.length, hashes=block_hashes(match))
            if operation == '/apply_delta':
                if not isinstance(data.get('known'), dict): raise ValueError('expected block digests')
                match = restore_delta(data.get('delta'), cache.match(tokens), data['known'])
                if match.length != len(tokens): raise ValueError('delta checkpoint length mismatch')
                cache.put(tokens, match.full, match.recurrent)
                if cache.match(tokens) != match: raise RuntimeError('checkpoint verification failed')
                return dict(length=match.length)
            if operation == '/match': return encode_match(cache.match(tokens))
            if operation == '/match_length': return dict(length=cache.match_length(tokens))
            if operation == '/delete': return dict(deleted=cache.delete(tokens))
            if operation == '/prepare_delta':
                return self.prepare_delta(tokens, data.get('known_blocks'), data.get('budget_bytes'), data.get('rate_limit', False))
            state = decode_match(data.get('state'))
            if state.length != len(tokens): raise ValueError('state length mismatch')
            cache.put(tokens, state.full, state.recurrent); return dict(stored=True)
        if operation == '/stage': return dict(handle=cache.stage_transfer(_decode(data.get('payload'))))
        handle = data.get('handle')
        if operation in ('/transfer', '/complete', '/abort'):
            if not isinstance(handle, str) or not 0 < len(handle) <= 128: raise ValueError('invalid handle')
            if operation == '/transfer': return dict(payload=base64.b64encode(cache.transfer_payload(handle)).decode('ascii'))
            if operation == '/complete': cache.complete_transfer(handle, data.get('acknowledged'))
            else: cache.abort_transfer(handle)
            return dict(released=True)
        raise ValueError('unsupported operation')

    def prepare_delta(self, tokens, known_blocks, budget_bytes, rate_limit=False):
        if type(rate_limit) is not bool: raise ValueError('rate_limit must be boolean')
        return prepare_delta(self.cache, tokens, known_blocks, budget_bytes,
                             self.rate_limiter if rate_limit else None, self.max_body_bytes)

    def start(self):
        if self._thread is not None: raise RuntimeError('node already started')
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={'poll_interval': .05}, daemon=True)
        self._thread.start(); return self

    def close(self):
        if self._thread is not None:
            self._server.shutdown(); self._thread.join(); self._thread = None
        self._server.server_close()

    def __enter__(self): return self.start()
    def __exit__(self, *exc): self.close()


class CacheClient:
    def __init__(self, url, namespace='default', token=None, timeout=15, max_body_bytes=64 * 1024 * 1024):
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('invalid cache node URL')
        if parsed.query or parsed.fragment or parsed.path not in ('', '/'):
            raise ValueError('cache node URL must be an origin')
        count(max_body_bytes, 'max_body_bytes', zero=False)
        if timeout <= 0: raise ValueError('timeout must be positive')
        self.url, self.namespace, self.token = url.rstrip('/'), namespace, token
        self.timeout, self.max_body_bytes = timeout, max_body_bytes
        info = self.stats()
        self.block_tokens, self.block_bytes = info['block_tokens'], info['block_bytes']
        self.capacity_blocks = info['capacity_blocks']

    def _request(self, operation, **data):
        body = _json_bytes(dict(namespace=self.namespace, **data))
        if len(body) > self.max_body_bytes: raise ValueError('request exceeds configured byte limit')
        headers = {'Content-Type': 'application/json'}
        if self.token is not None: headers['Authorization'] = 'Bearer ' + self.token
        request = Request(self.url + operation, body, headers, method='POST')
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read(self.max_body_bytes + 1)
        except HTTPError as error:
            raw = error.read(self.max_body_bytes + 1)
            if len(raw) > self.max_body_bytes: raise ValueError('error response exceeds configured byte limit') from error
            result = json.loads(raw)
            error_type = dict(ValueError=ValueError, TypeError=ValueError, KeyError=KeyError,
                              MemoryError=MemoryError, RuntimeError=RuntimeError, PermissionError=PermissionError)
            raise error_type.get(result.get('error'), RuntimeError)(result.get('message', 'cache request failed')) from error
        except (URLError, TimeoutError, ConnectionError) as error:
            raise ConnectionError(f'cache node unavailable: {self.url}') from error
        if len(raw) > self.max_body_bytes: raise ValueError('response exceeds configured byte limit')
        return json.loads(raw)

    def stats(self): return self._request('/info')
    @property
    def used_blocks(self): return self.stats()['used_blocks']
    def inventory(self): return self._request('/inventory')
    def match(self, tokens): return decode_match(self._request('/match', tokens=list(tokens)))
    def match_length(self, tokens): return self._request('/match_length', tokens=list(tokens))['length']
    def prepare_delta(self, tokens, known_blocks, budget_bytes, rate_limit=False):
        return self._request('/prepare_delta', tokens=list(tokens), known_blocks=known_blocks,
                             budget_bytes=budget_bytes, rate_limit=rate_limit)
    def put(self, tokens, full, recurrent):
        tokens = list(tokens)
        self._request('/put', tokens=tokens, state=encode_match(Match(len(tokens), full, recurrent)))
    def known(self, tokens):
        return self._request('/known', tokens=list(tokens))
    def apply_delta(self, tokens, delta, known):
        return self._request('/apply_delta', tokens=list(tokens), delta=delta, known=known)['length']
    def delete(self, tokens): return self._request('/delete', tokens=list(tokens))['deleted']
    def stage_transfer(self, payload):
        return self._request('/stage', payload=base64.b64encode(payload).decode('ascii'))['handle']
    def transfer_payload(self, handle): return _decode(self._request('/transfer', handle=handle)['payload'])
    def complete_transfer(self, handle, acknowledged):
        self._request('/complete', handle=handle, acknowledged=acknowledged)
    def abort_transfer(self, handle): self._request('/abort', handle=handle)
