"""A loopback bandwidth reservation clock shared by independent processes."""
from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import math
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

_MAX_BYTES = 4 * 1024**3
_MAX_DELAY = 24 * 60 * 60


def _bandwidth(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 10**15 or not math.isfinite(value):
        raise ValueError('bandwidth must be finite and between 0 and 10^15 bytes/s')
    return float(value)


def _token(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 256 or any(not 33 <= ord(c) <= 126 for c in value):
        raise ValueError('token must contain 1..256 printable ASCII characters without whitespace')
    return value


def _size(value):
    if type(value) is not int or not 0 <= value <= _MAX_BYTES:
        raise ValueError('bytes must be an integer between 0 and 4 GiB')
    return value


class RateService:
    """Reserve aggregate egress bytes using one monotonic completion clock.

    The service binds only loopback and authenticates every reservation. Changing
    bandwidth affects new reservations; already allocated slots are preserved.
    Outstanding reservations are bounded even when clients stop after reserving.
    """

    def __init__(self, bandwidth_bytes_s, token: str, *, max_pending: int = 4096):
        self._bandwidth = _bandwidth(bandwidth_bytes_s)
        self._token = _token(token)
        if type(max_pending) is not int or not 1 <= max_pending <= 4096:
            raise ValueError('max_pending must be an integer between 1 and 4096')
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._next = 0.0
        self._pending = deque()
        self._bytes = self._reservations = self._rejected = 0
        self._server = self._thread = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError('rate service is not running')
        return f'http://127.0.0.1:{self._server.server_port}/reserve'

    def set_bandwidth(self, bandwidth_bytes_s):
        value = _bandwidth(bandwidth_bytes_s)
        with self._lock:
            self._bandwidth = value

    def _prune(self, now):
        while self._pending and self._pending[0] <= now:
            self._pending.popleft()

    def stats(self) -> dict:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            return {'bytes': self._bytes, 'reservations': self._reservations,
                    'rejected': self._rejected, 'bandwidth_bytes_s': self._bandwidth,
                    'pending_reservations': len(self._pending),
                    'queued_seconds': max(0.0, self._next - now)}

    def _reserve(self, size):
        _size(size)
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            if self._bandwidth == 0:
                raise ConnectionError('bandwidth is unavailable')
            if not size:
                return 0.0
            if len(self._pending) >= self._max_pending:
                raise ConnectionError('bandwidth reservation queue is full')
            finish = max(now, self._next) + size / self._bandwidth
            if not math.isfinite(finish) or finish - now > _MAX_DELAY:
                raise ValueError('reservation would exceed the 24 hour queue limit')
            self._next = finish
            self._pending.append(finish)
            self._bytes += size
            self._reservations += 1
            return finish - now

    def __enter__(self):
        if self._server is not None:
            raise RuntimeError('rate service is already running')
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def setup(self):
                super().setup()
                self.connection.settimeout(3)

            def log_message(self, *args):
                pass

            def reply(self, status, body):
                data = json.dumps(body, separators=(',', ':'), allow_nan=False).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def reject(self, status, message):
                with owner._lock:
                    owner._rejected += 1
                self.reply(status, {'error': message})

            def do_POST(self):
                if self.path != '/reserve':
                    self.reject(404, 'unknown endpoint')
                    return
                auth = self.headers.get('Authorization', '')
                if not hmac.compare_digest(auth.encode(), ('Bearer ' + owner._token).encode()):
                    self.reject(401, 'invalid rate service token')
                    return
                try:
                    if self.headers.get('Transfer-Encoding'):
                        raise ValueError('transfer encoding is unsupported')
                    length = int(self.headers.get('Content-Length', '0'))
                    if not 1 <= length <= 256:
                        raise ValueError('invalid body length')
                    body = json.loads(self.rfile.read(length))
                    if not isinstance(body, dict) or set(body) != {'bytes'}:
                        raise ValueError('expected exactly one bytes field')
                    delay = owner._reserve(body['bytes'])
                except (ValueError, TypeError, UnicodeError) as error:
                    self.reject(400, str(error))
                    return
                except ConnectionError as error:
                    self.reject(503, str(error))
                    return
                self.reply(200, {'delay_s': delay})

        self._server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        kwargs={'poll_interval': 0.02}, daemon=True)
        self._thread.start()
        return self

    def close(self):
        server, thread = self._server, self._thread
        if server is None:
            return
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        self._server = self._thread = None

    def __exit__(self, *args):
        self.close()


def rate_client(url: str, token: str, size: int, *, timeout: float = 5) -> float:
    """Reserve bytes and sleep in the caller; return the allocated delay.

    The server answers immediately, so network timeout never includes pacing.
    A stopped/zero-bandwidth service raises ConnectionError; invalid input or
    credentials raise ValueError. Reservations are never silently retried.
    """
    _token(token)
    _size(size)
    if not isinstance(url, str):
        raise ValueError('rate service URL must be a string')
    parsed = urlsplit(url)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username or parsed.password:
        raise ValueError('rate service URL must use loopback HTTP without credentials')
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('timeout must be positive and finite')
    request = Request(url, data=json.dumps({'bytes': size}).encode(),
                      headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'}, method='POST')
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            raw = response.read(4097)
    except HTTPError as error:
        message = 'rate service rejected request'
        try:
            message = json.loads(error.read(4096)).get('error', message)
        except (ValueError, UnicodeError):
            pass
        if error.code in (400, 401, 403, 404):
            raise ValueError(message) from error
        raise ConnectionError(message) from error
    except (URLError, OSError) as error:
        raise ConnectionError('rate service is unavailable') from error
    try:
        result = json.loads(raw)
        delay = result['delay_s']
        if len(raw) > 4096 or isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or not 0 <= delay <= _MAX_DELAY:
            raise ValueError('invalid rate response delay')
    except (ValueError, TypeError, KeyError) as error:
        raise ConnectionError('invalid rate service response') from error
    time.sleep(delay)
    return float(delay)
