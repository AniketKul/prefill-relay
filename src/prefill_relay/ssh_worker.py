"""Launch a private prefill worker through OpenSSH loopback forwarding."""
from contextlib import ExitStack
import json
from pathlib import Path
import platform
import queue
import re
import secrets
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
from urllib.parse import urlsplit

from .worker import WorkerProcess, call


def _port(value):
    if type(value) is not int or not 0 < value <= 65535:
        raise ValueError('invalid TCP port')
    return value


def _url_port(url):
    target = urlsplit(url)
    if target.scheme != 'http' or target.hostname != '127.0.0.1' or target.username or target.password:
        raise ValueError('expected an IPv4 loopback HTTP endpoint')
    return _port(target.port)


class SSHWorker:
    """A WorkerProcess-compatible remote prefill producer.

    Existing OpenSSH multiplexed sessions can supply authentication. Only
    forwards established for this worker are cancelled when it exits.
    """
    def __init__(self, config, *, host, project='prefill-relay', device=None, cache_dir=None):
        if not isinstance(host, str) or not re.fullmatch(r'(?:[\w.][\w.-]*@)?[\w.][\w.-]*', host):
            raise ValueError('expected an SSH host alias, optionally user@alias')
        if not str(project) or str(project).startswith('-'):
            raise ValueError('invalid remote project directory')
        self.config = config
        self.host = host
        self.project = str(project)
        self.device = device or config['device']
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10']
        self.stack = ExitStack()
        self.reverse_ports = {}
        self.forward_lock = threading.Lock()

    def control(self, operation, *args, check=True):
        result = subprocess.run([*self.ssh, '-O', operation, *args, self.host],
                                capture_output=True, text=True, timeout=15)
        # Some OpenSSH versions report a mux forward failure only on stderr.
        if check and (result.returncode or (operation in ('forward', 'cancel') and result.stderr.strip())):
            raise OSError(f'SSH {operation} failed: {result.stderr.strip()}')
        return result

    def _open_master(self):
        if self.control('check', check=False).returncode == 0:
            return
        directory = self.stack.enter_context(tempfile.TemporaryDirectory(prefix='relay-ssh-'))
        self.ssh += ['-S', str(Path(directory) / 'mux')]
        subprocess.run([*self.ssh, '-M', '-N', '-f', '-o', 'ControlPersist=no', self.host], check=True, timeout=30)
        self.stack.callback(self.control, 'exit', check=False)

    def _forward(self, direction, destination):
        _port(destination)
        for _ in range(8):
            if direction == '-L':
                with socket.socket() as reservation:
                    reservation.bind(('127.0.0.1', 0))
                    port = reservation.getsockname()[1]
            else:
                port = 32768 + secrets.randbelow(28232)
            spec = f'127.0.0.1:{port}:127.0.0.1:{destination}'
            try:
                self.control('forward', direction, spec)
            except OSError:
                continue
            self.stack.callback(self.control, 'cancel', direction, spec, check=False)
            return port
        raise ConnectionError('SSH could not establish the requested loopback port forward')

    def _reverse(self, destination):
        with self.forward_lock:
            if destination not in self.reverse_ports:
                self.reverse_ports[destination] = self._forward('-R', destination)
            return self.reverse_ports[destination]

    def __enter__(self):
        try:
            self._open_master()
            rate = self._reverse(_url_port(self.config['rate_url']))
            config = dict(self.config, device=self.device, rate_url=f'http://127.0.0.1:{rate}/reserve')
            config['cache_path'] = Path(config['cache_path']).name
            command = (f'cd {shlex.quote(self.project)} && '
                       'exec "$(command -v uv || echo "$HOME/.local/bin/uv")" '
                       'run --locked --no-sync python -m prefill_relay.ssh_worker')
            process = subprocess.Popen([*self.ssh, '-T', self.host, command],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self.stack.callback(self._close, process)
            process.stdin.write(json.dumps({'config': config, 'cache_dir': self.cache_dir}) + '\n')
            process.stdin.flush()
            ready = queue.Queue(maxsize=1)
            def read_ready():
                for line in process.stdout:
                    if line.startswith('RELAY_READY '):
                        ready.put(line[len('RELAY_READY '):])
                        return
                ready.put(None)
            threading.Thread(target=read_ready, daemon=True).start()
            try:
                line = ready.get(timeout=600)
            except queue.Empty as error:
                raise TimeoutError('remote worker startup timed out') from error
            if not line:
                raise ConnectionError('remote worker exited before announcing its endpoints')
            self.info = json.loads(line)
            for field in ('url', 'cache_url'):
                port = self._forward('-L', _url_port(self.info[field]))
                self.info[field] = f'http://127.0.0.1:{port}'
            # Remote nodes only produce. Its tensor receiver is never a coordinator-side destination.
            self.info.pop('receiver', None)
            return self
        except BaseException:
            self.stack.close()
            raise

    @staticmethod
    def _close(process):
        try:
            process.stdin.close()
            try:
                process.wait(timeout=35)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        finally:
            process.stdout.close()

    def call(self, operation, **data):
        if operation == 'prefill':
            host, port = data['destination']
            if host != '127.0.0.1':
                raise ValueError('remote handoff requires an IPv4 loopback TCP receiver')
            data['destination'] = [host, self._reverse(_port(port))]
        return call(self.info['url'], self.config['token'], operation, **data)

    def __exit__(self, *exc):
        self.stack.close()


def serve_stdin():
    launch = json.loads(sys.stdin.readline(65537))
    config = launch['config']
    filename = config['cache_path']
    if filename != Path(filename).name or filename in ('', '.', '..'):
        raise ValueError('invalid remote cache filename')
    with ExitStack() as stack:
        directory = launch['cache_dir']
        if directory is None:
            directory = stack.enter_context(tempfile.TemporaryDirectory(prefix='relay-cache-'))
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        config['cache_path'] = str(directory / filename)
        worker = stack.enter_context(WorkerProcess(config))
        ready = dict(worker.info, host_system=platform.system(), device=config['device'])
        print('RELAY_READY ' + json.dumps(ready), flush=True)
        # EOF follows both a normal close and loss of the SSH session.
        sys.stdin.buffer.read(1)


if __name__ == '__main__':
    serve_stdin()
