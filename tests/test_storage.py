from pathlib import Path
import sqlite3
import tempfile
import unittest

from prefill_relay.storage import PersistentHybridCache


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'cache.sqlite'

    def tearDown(self):
        self.temp.cleanup()

    def cache(self, **kw):
        return PersistentHybridCache(self.path, block_tokens=2, block_bytes=8,
                                     capacity_blocks=kw.pop('capacity_blocks', 16),
                                     namespace=kw.pop('namespace', 'model-a'), **kw)

    def test_recomputed_recurrent_variant_after_checkpoint_eviction(self):
        with self.cache(capacity_blocks=3) as cache:
            cache.put([1,2],{'full':(b'A',)},{'linear':b'old'})
            cache._db.execute('UPDATE blocks SET touched=0 WHERE data=?',(b'A',))
            transfer=cache.stage_transfer(b'x'*16)
            self.assertEqual(cache.match_length([1,2]),0)
            cache.complete_transfer(transfer,True)
            cache.put([1,2],{'full':(b'A',)},{'linear':b'new'})
            self.assertEqual(cache.match([1,2]).recurrent['linear'],b'new')
            self.assertLessEqual(cache.used_blocks,3)

    def test_restart_recovers_exact_checkpoints_and_shared_blocks(self):
        with self.cache(ram_bytes=8) as cache:
            cache.put([1, 2], {'full': (b'12',)}, {'linear': b'state2'})
            cache.put([1, 2, 3, 4], {'full': (b'12', b'34')}, {'linear': b'state4'})
            self.assertEqual(cache.used_blocks, 4)
            self.assertLessEqual(cache.stats()['ram_bytes'], 8)
        with self.cache(ram_bytes=8) as cache:
            self.assertEqual(cache.match_length([1, 2, 3, 4, 5]), 4)
            self.assertEqual(cache.stats()['ssd_reads'], 0)
            hit = cache.match([1, 2, 9, 9])
            self.assertEqual(hit.length, 2)
            self.assertEqual(hit.recurrent, {'linear': b'state2'})
            self.assertEqual(cache.match([1, 2, 3, 4, 5]).length, 4)
            self.assertGreaterEqual(cache.stats()['ssd_reads'], 4)
            self.assertLessEqual(cache.stats()['ram_bytes'], 8)

    def test_without_exact_recurrent_state_partial_prefix_is_not_reusable(self):
        with self.cache() as cache:
            cache.put([1, 2, 3, 4], {'full': (b'12', b'34')}, {'linear': b'state4'})
            self.assertEqual(cache.match([1, 2, 5, 6]).length, 0)
            self.assertEqual(cache.match([1, 2, 3]).length, 0)

    def test_partial_final_block_is_reusable_only_with_its_exact_checkpoint(self):
        with self.cache() as cache:
            cache.put([1, 2, 3], {'full': (b'12', b'3')}, {'linear': b'state3'})
            cache.put([1, 2, 3, 4], {'full': (b'12', b'34')}, {'linear': b'state4'})
            self.assertEqual(cache.used_blocks, 5)
            self.assertEqual(cache.match([1, 2, 3, 9]).length, 3)
            self.assertEqual(cache.match([1, 2, 9]).length, 0)
            self.assertEqual(cache.match([1, 2, 3, 4]).length, 4)
        with self.cache() as cache:
            self.assertEqual(cache.match_length([1, 2, 3]), 3)
            self.assertEqual(cache.match([1, 2, 3]).full, {'full': (b'12', b'3')})
            self.assertTrue(cache.delete([1, 2, 3]))
            self.assertEqual(cache.match([1, 2, 3, 9]).length, 0)

    def test_transfer_pins_share_capacity_and_ack_controls_reclaim(self):
        with self.cache(capacity_blocks=2) as cache:
            handle = cache.stage_transfer(b'0123456789')
            self.assertEqual(cache.used_blocks, 2)
            self.assertEqual(cache.transfer_payload(handle), b'0123456789')
            with self.assertRaises(MemoryError): cache.put([1, 2], {'f': (b'12',)}, {'r': b'2'})
            with self.assertRaises(ValueError): cache.complete_transfer(handle, acknowledged=False)
            self.assertEqual(cache.used_blocks, 2)
            cache.complete_transfer(handle, acknowledged=True)
            self.assertEqual(cache.used_blocks, 0)

    def test_restart_aborts_orphaned_transfers_and_preserves_prefixes(self):
        with self.cache() as cache:
            cache.put([1, 2], {'f': (b'12',)}, {'r': b'2'})
            cache.stage_transfer(b'orphaned')
        with self.cache() as cache:
            self.assertEqual(cache.stats()['transfer_blocks'], 0)
            self.assertEqual(cache.match([1, 2]).length, 2)
            self.assertEqual(cache.stats()['recovered_transfers'], 1)

    def test_lru_eviction_and_put_failure_do_not_destroy_existing_entry(self):
        with self.cache(capacity_blocks=4) as cache:
            cache.put([1, 2], {'f': (b'12',)}, {'r': b'2'})
            cache.put([3, 4], {'f': (b'34',)}, {'r': b'4'})
            cache.match([1, 2])
            cache.put([5, 6], {'f': (b'56',)}, {'r': b'6'})
            self.assertEqual(cache.match([3, 4]).length, 0)
            self.assertEqual(cache.match([1, 2]).length, 2)
            with self.assertRaises(MemoryError):
                cache.put([10, 11, 12, 13], {'f': (b'aa', b'bb')}, {'r': b'x' * 25})
            self.assertEqual(cache.match([1, 2]).length, 2)

    def test_corruption_is_fail_closed_and_namespace_geometry_is_checked(self):
        with self.cache(ram_bytes=0) as cache:
            cache.put([1, 2], {'f': (b'12',)}, {'r': b'2'})
            with sqlite3.connect(self.path) as db:
                db.execute('UPDATE blocks SET data=? WHERE id=(SELECT id FROM blocks LIMIT 1)', (b'bad',))
            self.assertEqual(cache.match([1, 2]).length, 0)
            self.assertEqual(cache.stats()['corrupt_blocks'], 1)
        with self.assertRaises(ValueError): self.cache(namespace='other-model')

    def test_second_writer_is_rejected(self):
        with self.cache():
            with self.assertRaises(RuntimeError): self.cache()

    def test_idempotent_schema_order_and_repair_of_corrupt_disk_bytes(self):
        with self.cache(ram_bytes=0) as cache:
            cache.put([1, 2], {'a': (b'12',), 'b': (b'34',)}, {'x': b'2', 'y': b'4'})
            cache.put([1, 2], {'b': (b'34',), 'a': (b'12',)}, {'y': b'4', 'x': b'2'})
            with sqlite3.connect(self.path) as db:
                db.execute('UPDATE blocks SET data=? WHERE id=(SELECT id FROM blocks LIMIT 1)', (b'bad',))
            cache.put([1, 2], {'a': (b'12',), 'b': (b'34',)}, {'x': b'2', 'y': b'4'})
        with self.cache(ram_bytes=0) as cache:
            self.assertEqual(cache.match([1, 2]).length, 2)

    def test_inventory_delete_and_idempotent_put(self):
        with self.cache() as cache:
            cache.put([1, 2], {'f': (b'12',)}, {'r': b'2'})
            cache.put([1, 2], {'f': (b'12',)}, {'r': b'2'})
            self.assertEqual(len(cache.inventory()), 1)
            self.assertEqual(cache.used_blocks, 2)
            with self.assertRaises(ValueError): cache.put([1, 2], {'f': (b'XX',)}, {'r': b'2'})
            self.assertEqual(cache.match([1, 2]).full['f'], (b'12',))
            self.assertTrue(cache.delete([1, 2]))
            self.assertFalse(cache.delete([1, 2]))
            self.assertEqual(cache.used_blocks, 0)

    def test_full_block_spans_multiple_units_and_shares_capacity_with_transfer(self):
        payload = b'x' * 20
        with self.cache(capacity_blocks=4, ram_bytes=8) as cache:
            cache.put([1, 2], {'f': (payload,)}, {'r': b'2'})
            self.assertEqual(cache.used_blocks, 4)  # ceil(20 / 8) + one recurrent unit.
            self.assertEqual(cache.stats()['prefix_blocks'], 4)
            self.assertLessEqual(cache.stats()['ram_bytes'], 8)
        with self.cache(capacity_blocks=4, ram_bytes=8) as cache:
            self.assertEqual(cache.match([1, 2]).full, {'f': (payload,)})
            self.assertEqual(cache.used_blocks, 4)
            handle = cache.stage_transfer(b'z' * 24)
            self.assertEqual(cache.stats()['transfer_blocks'], 3)
            self.assertLessEqual(cache.used_blocks, 4)
            self.assertEqual(cache.match([1, 2]).length, 0)
            cache.complete_transfer(handle, True)

    def test_large_shared_full_block_is_protected_by_units_during_eviction(self):
        payload = b'x' * 20
        with self.cache(capacity_blocks=5) as cache:
            cache.put([1, 2], {'f': (payload,)}, {'r': b'2'})
            cache.put([1, 2, 3, 4], {'f': (payload, b'34')}, {'r': b'4'})
            self.assertEqual(cache.used_blocks, 5)
            self.assertEqual(cache.match([1, 2, 3, 4]).full['f'][0], payload)
            self.assertEqual(cache.match([1, 2]).length, 0)
            with self.assertRaises(MemoryError):
                cache.put([1, 2, 3, 4, 5, 6], {'f': (payload, b'34', b'56')}, {'r': b'6'})
            self.assertEqual(cache.match([1, 2, 3, 4]).length, 4)
            with self.assertRaises(MemoryError): cache.put([8, 9], {'f': (b'x' * 41,)}, {'r': b'9'})

    def test_legacy_single_unit_block_schema_is_migrated_on_restart(self):
        with self.cache() as cache:
            cache.put([1, 2], {'f': (b'12',)}, {'r': b'2'})
        with sqlite3.connect(self.path) as db:
            db.execute('DROP INDEX IF EXISTS blocks_units')
            db.execute('ALTER TABLE blocks DROP COLUMN units')
        with self.cache() as cache:
            self.assertEqual(cache.used_blocks, 2)
            self.assertEqual(cache.match([1, 2]).length, 2)

    def test_branch_checkpoints_keep_numeric_variants_and_dedupe_identical_blocks(self):
        with self.cache() as cache:
            cache.put([1, 2, 3, 4], {'f': (b'first-A', b'34')}, {'r': b's4'})
            cache.put([1, 2, 7, 8], {'f': (b'first-B', b'78')}, {'r': b's8'})
            self.assertEqual(cache.used_blocks, 6)
            cache.put([1, 2, 9, 10], {'f': (b'first-A', b'90')}, {'r': b's10'})
            self.assertEqual(cache.used_blocks, 8)
            self.assertEqual(cache.match([1, 2, 3, 4]).full['f'], (b'first-A', b'34'))
            self.assertEqual(cache.match([1, 2, 7, 8]).full['f'], (b'first-B', b'78'))
            self.assertEqual(cache.match([1, 2, 9, 10]).full['f'], (b'first-A', b'90'))
            with self.assertRaises(ValueError):
                cache.put([1, 2, 3, 4], {'f': (b'changed', b'34')}, {'r': b's4'})
            self.assertEqual(cache.used_blocks, 8)
        with self.cache() as cache:
            self.assertEqual(cache.match([1, 2, 3, 4]).full['f'], (b'first-A', b'34'))
            self.assertEqual(cache.match([1, 2, 7, 8]).full['f'], (b'first-B', b'78'))

    def test_full_capacity_cannot_evict_and_replace_a_conflicting_exact_checkpoint(self):
        with self.cache(capacity_blocks=2) as cache:
            cache.put([1, 2], {'f': (b'old',)}, {'r': b'2'})
            with self.assertRaises(ValueError): cache.put([1, 2], {'f': (b'new',)}, {'r': b'2'})
            self.assertEqual(cache.match([1, 2]).full['f'], (b'old',))
            self.assertEqual(cache.used_blocks, 2)


if __name__ == '__main__': unittest.main()
