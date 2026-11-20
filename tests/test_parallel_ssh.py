import os
import pwd
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from prefill_relay.ssh_worker import SSHWorker
from prefill_relay.transport import Receiver, StreamSender, assemble_chunks, ssh_stream_connector


class ParallelSSHTests(unittest.TestCase):
    def test_rejects_unexpected_forward_destinations(self):
        for ports in ([], [0], [123,123], [124], [123,65536], [123,124,125]):
            with self.assertRaises(ValueError):
                ssh_stream_connector(('127.0.0.1',123), ports, 2)
        with self.assertRaises(ValueError):
            ssh_stream_connector(('192.0.2.1',123), [123], 2)

    @unittest.skipUnless(os.environ.get('PREFILL_RELAY_TEST_SSHD') == '1',
                         'set PREFILL_RELAY_TEST_SSHD=1 to use an isolated loopback sshd')
    def test_two_ssh_transports_carry_independent_streams(self):
        with tempfile.TemporaryDirectory(prefix='relay-sshd-', dir='/tmp') as path:
            directory = Path(path)
            client_key, host_key = directory/'client', directory/'host'
            for key in (client_key, host_key):
                subprocess.run(['ssh-keygen','-q','-N','','-t','ed25519','-f',str(key)],check=True)
            with socket.socket() as port_probe:
                port_probe.bind(('127.0.0.1', 0))
                port = port_probe.getsockname()[1]
            (directory/'known_hosts').write_text(f'[127.0.0.1]:{port} ' + host_key.with_suffix('.pub').read_text())
            client_config = directory/'client-config'
            client_config.write_text(f'''Host loopback-fixture
  HostName 127.0.0.1
  Port {port}
  IdentityFile {client_key}
  IdentitiesOnly yes
  BatchMode yes
  StrictHostKeyChecking yes
  UserKnownHostsFile {directory/'known_hosts'}
''')
            config = directory/'sshd-config'
            config.write_text(f'''ListenAddress 127.0.0.1
Port {port}
HostKey {host_key}
PidFile {directory/'sshd.pid'}
AuthorizedKeysFile {client_key}.pub
AllowUsers {pwd.getpwuid(os.getuid()).pw_name}
PasswordAuthentication no
KbdInteractiveAuthentication no
UsePAM no
StrictModes no
AllowAgentForwarding no
X11Forwarding no
GatewayPorts no
ForceCommand false
LogLevel VERBOSE
''')
            daemon = shutil.which('sshd') or '/usr/sbin/sshd'
            with open(directory/'log','w') as log:
                sshd = subprocess.Popen([daemon,'-D','-e','-f',str(config)],stderr=log)
                try:
                    deadline = time.monotonic()+4
                    while 'Server listening' not in (directory/'log').read_text() and time.monotonic()<deadline:
                        if sshd.poll() is not None:
                            raise RuntimeError((directory/'log').read_text())
                        time.sleep(.01)
                    worker = SSHWorker({'device': 'cpu', 'connections': 2, 'token': 'fixture'},
                                       host='loopback-fixture', streams=2)
                    worker.ssh = ['ssh','-F',str(client_config)]
                    with Receiver(timeout=3) as decoder, worker.stack:
                        worker._open_master()
                        worker.info = {'url': 'http://127.0.0.1'}
                        with patch('prefill_relay.ssh_worker.call') as rpc:
                            worker.call('prefill', destination=decoder.address)
                        handoff = rpc.call_args.kwargs
                        ports = handoff['ssh_stream_ports']
                        self.assertEqual(len(ports), 2)
                        connect = ssh_stream_connector(handoff['destination'], ports, 2)
                        both_open = threading.Barrier(2)
                        def observed_connect(address, timeout):
                            sock = connect(address, timeout)
                            both_open.wait(timeout=3)
                            return sock
                        payload = bytes(range(64))*16
                        with StreamSender(('127.0.0.1',ports[0]), 'ssh-test', 2, timeout=3,
                                          connect=observed_connect) as sender:
                            sender.submit_blob('state',payload,chunk_bytes=32)
                        self.assertEqual(assemble_chunks(decoder.take('ssh-test')), {'state':payload})
                        self.assertEqual(sender.stats['frames_acked'],32)
                        self.assertEqual(sender.stats['connections_opened'],2)
                    self.assertEqual((directory/'log').read_text().count('Accepted publickey'), 2)
                finally:
                    sshd.terminate()
                    sshd.wait(timeout=5)
