import importlib.util
import os
from pathlib import Path
import secrets
import tempfile
import unittest

from prefill_relay.ssh_worker import SSHWorker, _url_port


class SSHConfigurationTests(unittest.TestCase):
    def test_endpoint_and_host_validation(self):
        for url in ('http://localhost:20', 'http://example.org:80',
                    'https://127.0.0.1:123', 'http://foo@127.0.0.1:123',
                    'http://127.0.0.1:0', 'http://127.0.0.1:65536'):
            with self.assertRaises(ValueError):
                _url_port(url)
        for host in ('-oToken', 'hostname x', 'foo;ls', 'a@b@c'):
            with self.assertRaises(ValueError):
                SSHWorker({}, host=host)


@unittest.skipUnless(os.environ.get('PREFILL_RELAY_SSH_HOST') and importlib.util.find_spec('torch'),
                     'set PREFILL_RELAY_SSH_HOST to run the remote worker check')
class RemoteHandoffTests(unittest.TestCase):
    def test_real_remote_prefill_and_local_decode(self):
        from prefill_relay.cache_service import CacheClient
        from prefill_relay.rate_limit import RateService
        from prefill_relay.worker import WorkerProcess

        token = secrets.token_urlsafe(32)
        with tempfile.TemporaryDirectory() as directory, RateService(100*2**20, token) as rate:
            config = dict(backend='tiny', device='cpu', token=token, role='decode',
                          cache_path=str(Path(directory)/'decode.sqlite'), rate_url=rate.url)
            remote = SSHWorker(dict(config, role='prefill', cache_path='prefill.sqlite'),
                               host=os.environ['PREFILL_RELAY_SSH_HOST'],
                               project=os.environ.get('PREFILL_RELAY_SSH_PROJECT', 'prefill-relay'),
                               device=os.environ.get('PREFILL_RELAY_REMOTE_DEVICE', 'cpu'))
            tokens = [(i * 17 + 3) % 256 for i in range(67)]
            with WorkerProcess(config) as decoder, remote as producer:
                self.assertEqual(decoder.info['namespace'], producer.info['namespace'])
                reference = decoder.call('reference', tokens=tokens, output_tokens=4)
                client = CacheClient(producer.info['cache_url'], producer.info['namespace'], token)
                for prompt in (tokens[:33], tokens):
                    key = f'request-{len(prompt)}'
                    base = decoder.call('prepare', request=key, tokens=prompt)['prefix_tokens']
                    prefill = producer.call('prefill', request=key, tokens=prompt,
                                            destination=decoder.info['receiver'],
                                            destination_prefix=base, rate_limit=True)
                    decoder.call('consume', request=key, first_token=prefill['first_token'])
                    result = decoder.call('decode', request=key, output_tokens=4)
                    self.assertEqual(client.match_length(prompt), len(prompt))
                self.assertEqual(base, 33)
                self.assertEqual(prefill['prefix_tokens'], 33)
                self.assertEqual(result['generated_tokens'], reference['generated_tokens'])
                self.assertGreater(rate.stats()['bytes'], 0)
                self.assertEqual(producer.call('stats')['transfers'], 0)
