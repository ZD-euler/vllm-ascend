# SPDX-License-Identifier: Apache-2.0
"""Run the actual lease refresh method with a deterministic SDK boundary."""

import ast
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def load_worker():
    path = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py"
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KVPoolWorker")
    method = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_load_gvas")
    cls = ast.ClassDef(name="Worker", bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    scope = dict(
        np=np,
        logger=Mock(),
        MEMCACHE_UNMATCHED_STATE=777,
        LAYERWISE_READ_LEASE_TTL_MS=1000,
        PARTIAL_LEASE_RETRY_COUNT=3,
        PARTIAL_LEASE_RETRY_INTERVAL_S=0.1,
        time=NS(sleep=lambda _: None),
        get_block_hashes=lambda values, *_: values,
        block_hash_to_str=str,
        get_partial_block_index=lambda tokens, block, *_: tokens // block if tokens % block else None,
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["Worker"]


class Store:
    def __init__(self, missing=(), failed=(), refreshed_missing=(), short=False):
        self.ids = {}
        self.calls = []
        self.missing, self.failed = set(missing), set(failed)
        self.refreshed_missing, self.short = set(refreshed_missing), short

    def value(self, key, refreshed):
        self.ids.setdefault(key, len(self.ids) + 1)
        return (2000000 if refreshed else 1000000) + self.ids[key]

    def batch_get_key_info(self, keys, for_load=False):
        self.calls.append(("info", tuple(keys), for_load))
        result = []
        for key in keys:
            absent = key in (self.missing if for_load else self.refreshed_missing)
            result.append(
                None
                if absent
                else NS(size=lambda: 4096, gva_list=lambda k=key, refresh=not for_load: [self.value(k, refresh)])
            )
        return result[:-1] if self.short and not for_load else result

    def batch_add_lease(self, keys, _ttl):
        self.calls.append(("add", tuple(keys)))
        return [7 if k in self.failed else 0 for k in keys]

    def batch_remove_lease(self, keys):
        self.calls.append(("remove", tuple(keys)))
        return 0


def request(hashes, history=None):
    history = len(hashes) * 128 if history is None else history
    return NS(
        req_id="r",
        block_hashes=hashes,
        load_spec=NS(
            can_load=True, kvpool_cached_tokens=history, kvpool_store_skip_tokens=history, vllm_cached_tokens=0
        ),
        block_ids_by_group_np=[np.arange(1, len(hashes) + 2, dtype=np.int64)],
        block_ids_np=None,
    )


def worker(store=None, selected=True):
    value = load_worker()()
    value.use_layerwise_transfer = value.layerwise_offload = True
    value.num_kv_cache_groups, value.cacheable_group_ids = 1, [0]
    value.grouped_block_size, value.hash_block_size = [128], 128
    value._compute_reachable_load_masks = lambda *_: None
    value._make_layerwise_full_key = lambda group, name: "full-" + name
    value._make_layerwise_partial_key = lambda *_: "partial"
    value.m_store = store or Store()
    value.selected_kv_source = NS(record_leased_infos=Mock()) if selected else None
    value._invalid_block_ids, value._invalid_block_ids_lock = set(), threading.Lock()
    return value


class LeaseRefreshTest(unittest.TestCase):
    def test_surviving_keys_keep_their_original_positions(self):
        for selected in (False, True):
            with self.subTest(selected=selected):
                backend = Store(missing={"full-b"}, failed={"full-c"})
                value, req = worker(backend, selected), request(["a", "b", "c", "d"])
                value._prepare_load_gvas([req])
                expected = [backend.value("full-a", True), 0, 0, backend.value("full-d", True), 0]
                np.testing.assert_array_equal(req.load_block_gvas_by_group_np[0], expected)
                self.assertEqual(value._invalid_block_ids, {2, 3})
                self.assertEqual(req.load_keys, ["full-a", "full-d"])

    def test_duplicate_keys_preserve_original_first_occurrence_behavior(self):
        value, req = worker(), request(["a", "b", "a"])
        value._prepare_load_gvas([req])
        expected = [
            value.m_store.value("full-a", True),
            value.m_store.value("full-b", True),
            value.m_store.value("full-a", False),
            0,
        ]
        np.testing.assert_array_equal(req.load_block_gvas_by_group_np[0], expected)
        self.assertEqual(req.load_keys, ["full-a", "full-b", "full-a"])

    def test_partial_key_refresh_is_kept_out_of_full_page_vector(self):
        value, req = worker(), request(["a", "b", "tail"], history=273)
        value._prepare_load_gvas([req])
        self.assertEqual(req.partial_load_gva_per_group, [value.m_store.value("partial", True)])
        self.assertEqual(req.load_block_gvas_by_group_np[0][2], 0)
        self.assertEqual(req.load_keys, ["full-a", "full-b", "partial"])

    def test_refresh_failure_reports_the_same_physical_block(self):
        value, req = worker(Store(refreshed_missing={"full-b"})), request(["a", "b", "c"])
        value._prepare_load_gvas([req])
        self.assertEqual(value._invalid_block_ids, {2})
        self.assertEqual(req.load_block_gvas_by_group_np[0][1], 0)

    def test_short_response_releases_leases_before_raising(self):
        value, req = worker(Store(short=True)), request(["a", "b"])
        with self.assertRaisesRegex(RuntimeError, "response length mismatch"):
            value._prepare_load_gvas([req])
        self.assertEqual(value.m_store.calls[-1], ("remove", ("full-a", "full-b")))

    def test_unexpected_refreshed_key_fails_closed(self):
        class MutatingStore(Store):
            def batch_get_key_info(self, keys, for_load=False):
                infos = super().batch_get_key_info(keys, for_load)
                if not for_load:
                    keys[0] = "unexpected-key"
                return infos

        for selected in (False, True):
            with self.subTest(selected=selected):
                backend = MutatingStore()
                value, req = worker(backend, selected), request(["a", "b"])
                with self.assertRaisesRegex(KeyError, "unexpected-key"):
                    value._prepare_load_gvas([req])
                self.assertEqual(
                    backend.calls,
                    [
                        ("info", ("full-a", "full-b"), True),
                        ("add", ("full-a", "full-b")),
                        ("info", ("full-a", "full-b"), False),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
