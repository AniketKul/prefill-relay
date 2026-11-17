import base64
import hashlib
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from prefill_relay.cache_service import CacheClient, CacheNode
from prefill_relay.directory import GlobalDirectory
from prefill_relay.storage import PersistentHybridCache


def node_process(path, pipe, capacity_blocks=256, block_tokens=2, block_bytes=16, rate_log=None):
    try:
        def limiter(size):
            with open(rate_log, 'a') as log: log.write(str(size) + '\n')
        with PersistentHybridCache(path, block_tokens=block_tokens, block_bytes=block_bytes, capacity_blocks=capacity_blocks,
                                   namespace='model-a', ram_bytes=16) as cache:
            with CacheNode(cache, token='test-authentication', max_body_bytes=8192,
                           rate_limiter=limiter if rate_log else None) as node:
                pipe.send(node.url)
                pipe.recv()
    finally:
        pipe.close()


def crash_process(path):
    import os
    cache = PersistentHybridCache(path, block_tokens=2, block_bytes=16, capacity_blocks=256,
                                  namespace='model-a', ram_bytes=16)
    cache.put([1, 2], {'full': (b'full12',)}, {'recurrent': b'state12'})
    cache.stage_transfer(b'pending-state')
    os._exit(0)


class CacheServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ctx = multiprocessing.get_context('spawn')
        self.children = []

    def tearDown(self):
        for process, pipe in self.children:
            if process.is_alive():
                try: pipe.send('stop')
                except BrokenPipeError: pass
                process.join(5)
                if process.is_alive(): process.kill(); process.join()
            pipe.close()
        self.temp.cleanup()

    def start(self, name, capacity_blocks=256, block_tokens=2, block_bytes=16, rate_log=None):
        parent, child = self.ctx.Pipe()
        process = self.ctx.Process(target=node_process,
                                   args=(str(Path(self.temp.name) / name), child, capacity_blocks,
                                         block_tokens, block_bytes, rate_log))
        process.start(); child.close()
        self.children.append((process, parent))
        self.assertTrue(parent.poll(15), 'node did not start')
        url = parent.recv()
        return CacheClient(url, namespace='model-a', token='test-authentication', timeout=2)

    def store(self, client):
        client.put([1, 2], {'full': (b'full12',)}, {'recurrent': b'state12'})
        client.put([1, 2, 3, 4], {'full': (b'full12', b'full34')}, {'recurrent': b'state34'})

    def test_real_process_auth_namespace_bounds_and_roundtrip(self):
        client = self.start('node-a.db')
        self.store(client)
        self.assertEqual(client.match_length([1, 2, 3, 4, 5]), 4)
        self.assertEqual(client.match([1, 2, 3, 4, 5]).length, 4)
        self.assertEqual(client.match([1, 2, 7]).recurrent, {'recurrent': b'state12'})
        self.assertEqual(len(client.inventory()), 2)
        with self.assertRaises(PermissionError): CacheClient(client.url, 'model-a', token='wrong')
        with self.assertRaises(ValueError): CacheClient(client.url, 'other', token='test-authentication')
        with self.assertRaises(ValueError): client.stage_transfer(b'x' * 9000)
        with self.assertRaises(ValueError): client.match([-1])
        handle = client.stage_transfer(b'network-payload')
        self.assertEqual(client.transfer_payload(handle), b'network-payload')
        with self.assertRaises(ValueError): client.complete_transfer(handle, False)
        self.assertGreater(client.stats()['transfer_blocks'], 0)
        client.complete_transfer(handle, True)
        self.assertEqual(client.stats()['transfer_blocks'], 0)
        self.assertTrue(client.delete([1, 2, 3, 4]))
        self.assertEqual(client.match([1, 2, 3, 4]).length, 2)

    def test_live_directory_placement_migration_and_same_cluster_rebalance(self):
        first, second, third = (self.start(name) for name in ('a.db', 'b.db', 'c.db'))
        self.store(first)
        directory = GlobalDirectory()
        directory.register('west', 'a', first); directory.register('west', 'b', second)
        directory.register('east', 'c', third)
        self.assertEqual(directory.best('west', 'model-a', [1, 2, 3, 4]), ('a', 4))
        self.assertEqual(directory.best('west', 'wrong-model', [1, 2]), (None, 0))
        denied = directory.migrate(('west', 'a'), ('west', 'b'), 'model-a', [1, 2, 3, 4], 1)
        self.assertEqual(denied.status, 'budget_exceeded')
        self.assertEqual(second.used_blocks, 0)
        receipt = directory.rebalance('model-a', [1, 2, 3, 4],
                                      {('west', 'a'): 10, ('west', 'b'): 1, ('east', 'c'): 0}, 4096)
        self.assertEqual(receipt.status, 'transferred')
        self.assertEqual(receipt.target, ('west', 'b'))
        self.assertEqual(directory.best('west', 'model-a', [1, 2, 3, 4]), ('b', 4))
        self.assertEqual(directory.fetch('west', 'b', 'model-a', [1, 2, 3, 4]), first.match([1, 2, 3, 4]))
        self.assertEqual(first.stats()['transfer_blocks'], 0)
        self.assertEqual(third.used_blocks, 0)
        self.assertEqual(directory.rebalance('model-a', [1, 2, 3, 4], budget_bytes=4096).status, 'balanced')
        cross = directory.rebalance('model-a', [1, 2, 3, 4], budget_bytes=4096, allow_cross_cluster=True)
        self.assertEqual(cross.target, ('east', 'c'))
        self.assertEqual(third.match([1, 2, 3, 4]).length, 4)
        self.children[0][0].terminate(); self.children[0][0].join(5)
        self.assertEqual(directory.best('west', 'model-a', [1, 2, 3, 4]), ('b', 4))

    def test_abrupt_process_exit_recovers_committed_data_and_releases_orphan_pins(self):
        path = str(Path(self.temp.name) / 'crashed.db')
        process = self.ctx.Process(target=crash_process, args=(path,))
        process.start(); process.join(15)
        self.assertEqual(process.exitcode, 0)
        client = self.start('crashed.db')
        self.assertEqual(client.match([1, 2]).recurrent, {'recurrent': b'state12'})
        self.assertEqual(client.stats()['transfer_blocks'], 0)
        self.assertEqual(client.stats()['recovered_transfers'], 1)

    def test_listener_requires_authentication_off_loopback(self):
        with PersistentHybridCache(Path(self.temp.name) / 'local.db') as cache:
            with self.assertRaises(ValueError): CacheNode(cache, host='0.0.0.0')

    def test_receiver_capacity_failure_releases_sender_pins(self):
        first, second = self.start('source.db'), self.start('small.db', capacity_blocks=1)
        self.store(first)
        directory = GlobalDirectory()
        directory.register('west', 'a', first); directory.register('west', 'b', second)
        with self.assertRaises(MemoryError):
            directory.migrate(('west', 'a'), ('west', 'b'), 'model-a', [1, 2, 3, 4], 4096)
        self.assertEqual(first.stats()['transfer_blocks'], 0)
        self.assertEqual(second.used_blocks, 0)
        self.assertEqual(first.match([1, 2, 3, 4]).length, 4)

    def test_delta_migration_16_to_32_sends_missing_blocks_and_rates_only_cross_cluster(self):
        rate_log = Path(self.temp.name) / 'rate.log'
        source = self.start('source.db', block_tokens=16, block_bytes=1024, rate_log=str(rate_log))
        cold = self.start('cold.db', block_tokens=16, block_bytes=1024)
        warm = self.start('warm.db', block_tokens=16, block_bytes=1024)
        tokens, first, second = list(range(32)), b'a' * 1024, b'b' * 1024
        source.put(tokens, {'f': (first, second)}, {'r': b'state-at-32'})
        warm.put(tokens[:16], {'f': (first,)}, {'r': b'state-at-16'})
        known = {'f': [hashlib.sha256(first).hexdigest()]}
        prepared = source.prepare_delta(tokens, known, 8192)
        self.assertIsNone(prepared['delta']['full']['f'][0])
        self.assertEqual(base64.b64decode(prepared['delta']['full']['f'][1]), second)
        self.assertEqual(base64.b64decode(prepared['delta']['recurrent']['r']), b'state-at-32')
        self.assertEqual(source.stats()['transfers'], 1)
        source.abort_transfer(prepared['handle'])
        warm_bytes = prepared['payload_bytes']
        denied = source.prepare_delta(tokens, {}, warm_bytes)
        self.assertEqual(denied['status'], 'budget_exceeded')
        self.assertIsNone(denied['handle'])
        self.assertIsNone(denied['delta'])
        self.assertEqual(source.stats()['transfers'], 0)
        directory = GlobalDirectory()
        directory.register('west', 'source', source)
        directory.register('west', 'cold', cold)
        directory.register('east', 'warm', warm)
        # Fail if the coordinator downloads, re-uploads, or re-fetches source payloads.
        with patch.object(source, 'match', side_effect=AssertionError('full source fetch')), \
             patch.object(source, 'stage_transfer', side_effect=AssertionError('source upload')), \
             patch.object(source, 'transfer_payload', side_effect=AssertionError('second source download')):
            cold_receipt = directory.migrate(('west', 'source'), ('west', 'cold'), 'model-a', tokens, 8192)
            self.assertFalse(rate_log.exists())
            warm_receipt = directory.migrate(('west', 'source'), ('east', 'warm'), 'model-a', tokens, warm_bytes)
        self.assertEqual(warm_receipt.bytes_transferred, warm_bytes)
        self.assertLess(warm_receipt.bytes_transferred, cold_receipt.bytes_transferred * .65)
        self.assertEqual(warm.match(tokens), cold.match(tokens))
        self.assertEqual(warm.match(tokens).recurrent, {'r': b'state-at-32'})
        self.assertEqual(rate_log.read_text().splitlines(), [str(warm_receipt.bytes_transferred)])
        self.assertEqual(source.stats()['transfers'], 0)

    def test_partial_block_hash_mismatch_sends_new_bytes_and_bad_hashes_are_rejected(self):
        source = self.start('partial.db')
        self.store(source)
        known = {'full': [hashlib.sha256(b'full12').hexdigest(), hashlib.sha256(b'partial3').hexdigest()]}
        prepared = source.prepare_delta([1, 2, 3, 4], known, 4096)
        self.assertIsNone(prepared['delta']['full']['full'][0])
        self.assertEqual(base64.b64decode(prepared['delta']['full']['full'][1]), b'full34')
        source.abort_transfer(prepared['handle'])
        with self.assertRaises(ValueError): source.prepare_delta([1, 2], {'full': ['bad-hash']}, 4096)
        self.assertEqual(source.stats()['transfers'], 0)

    def test_rate_limiter_failure_releases_local_staging_before_error_response(self):
        def unavailable(size): raise RuntimeError('rate service unavailable')
        with PersistentHybridCache(Path(self.temp.name) / 'rate-failure.db', block_tokens=2,
                                   block_bytes=16, capacity_blocks=256, namespace='model-a') as cache:
            cache.put([1, 2], {'f': (b'12',)}, {'r': b'2'})
            with CacheNode(cache, rate_limiter=unavailable) as node:
                client = CacheClient(node.url, namespace='model-a')
                with self.assertRaises(RuntimeError): client.prepare_delta([1, 2], {}, 4096, rate_limit=True)
                self.assertEqual(client.stats()['transfers'], 0)


if __name__ == '__main__': unittest.main()
