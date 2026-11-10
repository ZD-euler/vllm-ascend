# SPDX-License-Identifier: Apache-2.0
"""Compose real native GVA preparation, partial-page builder and selected source.

Only store RPCs and device upload/events use CPU leaves from the source fixture.
Production methods are extracted to avoid importing the unavailable NPU SDK.
"""

import ast
import builtins
import types
import unittest
from types import SimpleNamespace

import numpy as np
from test_selected_kv_source import (
    STORE,
    Event,
    Shared,
    finish_readers,
    fixture,
    load_methods,
    request,
    reset_native_tasks,
    scheduled_tokens,
)


def load_functions(filename, names, namespace):
    tree = ast.parse((STORE / filename).read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(STORE / filename), "exec"), namespace)


def strict_zip(*iterables, strict=False):
    # System CPU Torch is Python 3.9; retain the production strict=True check.
    if strict:
        lengths = [len(values) for values in iterables]
        if len(set(lengths)) > 1:
            raise ValueError("zip() arguments have different lengths")
    return builtins.zip(*iterables)


class NativePartialIntegrationTest(unittest.TestCase):
    def test_full_hit_keeps_native_full_boundary_locator(self):
        # A fully cached prompt recomputes its final token: Q=1, H=prompt-1.
        req = request(h=65663, q=1)
        selected, worker, metadata, _, exporter = fixture([req])
        req.load_spec.kvpool_cached_tokens = req.target_token_len - 1
        req.load_spec.kvpool_store_skip_tokens = req.target_token_len
        req.block_hashes = [f"full-{i}" for i in range(req.target_token_len // 128)]
        req.load_spec.vllm_cached_tokens = 0
        req.partial_load_gva_per_group = []
        worker.selected_kv_source = selected
        worker.cacheable_group_ids, worker.hash_block_size = [0], 128
        worker.independent_layers = []
        worker._step_keyinfo_cache, worker._step_lease_result = {}, {}
        worker._compute_reachable_load_masks = lambda *_: None
        worker._make_layerwise_full_key = lambda group, key: key
        worker._make_layerwise_partial_key = lambda *_: "unexpected-partial"
        infos = {}

        def key_info(keys, **kwargs):
            for key in keys:
                if key not in infos:
                    gva = (1 << 24) + len(infos) * (1 << 20)
                    infos[key] = SimpleNamespace(gva_list=lambda gva=gva: [gva], size=lambda: 1 << 24)
            return [infos[key] for key in keys]

        worker.m_store.batch_get_key_info = key_info
        worker.m_store.batch_add_lease = lambda keys, ttl: [0] * len(keys)
        namespace = {
            "np": np,
            "zip": strict_zip,
            "logger": SimpleNamespace(debug=lambda *a: None, warning=lambda *a: None),
            "LAYERWISE_READ_LEASE_TTL_MS": 1000,
            "MEMCACHE_UNMATCHED_STATE": -3101,
        }
        load_functions("metadata.py", {"get_partial_block_index", "get_block_hashes", "block_hash_to_str"}, namespace)
        pool = load_methods("pool_worker.py", "KVPoolWorker", {"_prepare_load_gvas"}, namespace)
        pool._prepare_load_gvas(worker, [req])
        self.assertEqual(req.partial_load_gva_per_group, [0])
        self.assertGreater(req.load_block_gvas_by_group_np[0][512], 0)
        # Preserve actual native locators, including the aligned full-hit page.
        reset_native_tasks(worker, metadata.requests)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertIsNotNone(step)
        self.assertTrue(step.requests[0].selected)
        self.assertEqual(int(step.page_bases("layer.0.main")[513]), 0)
        self.assertTrue(np.all(step.page_bases("layer.0.main")[1:513] > 0))
        self.assertEqual(len(exporter.calls[0][0]), 512)
        for tasks in worker.layer_load_tasks:
            shared = tasks[0].shared_block_data
            self.assertTrue(shared.main_load_filter.keep_main[shared.block_ids_arr == 513].all())
        finish_readers(step)
        self.assertTrue(step.finish(Event()))
        selected.close()

    def test_real_native_partial_prepare_then_selected_mixed_step(self):
        selected_req = request(h=65537, q=125)
        native_req = request("native", h=65536, q=4096, first=1000)
        selected, worker, metadata, _, exporter = fixture([selected_req, native_req])
        namespace = {
            "np": np,
            "zip": strict_zip,
            "logger": SimpleNamespace(debug=lambda *a: None, info=lambda *a: None, warning=lambda *a: None),
            "LAYERWISE_READ_LEASE_TTL_MS": 1000,
            "MEMCACHE_UNMATCHED_STATE": -3101,
            "LayerBlockRange": SimpleNamespace,
            "LayerTransferTask": SimpleNamespace,
            "SharedBlockData": Shared,
        }
        load_functions(
            "metadata.py",
            {
                "get_block_hashes",
                "get_partial_block_index",
                "masked_block_runs",
                "get_group_block_size",
                "block_hash_to_str",
            },
            namespace,
        )
        load_functions("backend/memcache_backend.py", {"make_full_key", "make_partial_key"}, namespace)
        pool = load_methods(
            "pool_worker.py",
            "KVPoolWorker",
            {
                "_prepare_load_gvas",
                "_process_load_for_layer_batch",
                "_masked_block_runs_cached",
                "_compute_reachable_load_masks",
                "_make_layerwise_full_key",
                "_make_layerwise_partial_key",
            },
            namespace,
        )
        for name in vars(pool):
            if name.startswith("_") and not name.startswith("__"):
                setattr(worker, name, types.MethodType(getattr(pool, name), worker))
        worker.cacheable_group_ids = [0]
        worker.hash_block_size = 128
        worker.cache_coordinator = None
        worker.independent_layers = []
        worker._masked_runs_cache = {}
        worker.model_name, worker.pp_rank = "model", 0
        worker.metadata = [SimpleNamespace(head_or_tp_rank=0)]
        worker.layerwise_protocol = SimpleNamespace(make_partial_key=namespace["make_partial_key"])
        worker.layerwise_keys = SimpleNamespace(
            make_full_key=lambda group, block_hash, rank, pp: namespace["make_full_key"](
                "model", group, block_hash, rank, 1, pp
            )
        )
        worker._step_keyinfo_cache, worker._step_lease_result = {}, {}
        selected._leased_infos.clear()
        worker.selected_kv_source = selected
        infos = {}

        def key_info(keys, **kwargs):
            for key in keys:
                if key not in infos:
                    gva = (1 << 24) + len(infos) * (1 << 20)
                    infos[key] = SimpleNamespace(gva_list=lambda value=gva: [value], size=lambda: 1 << 24)
            return [infos[key] for key in keys]

        worker.m_store.batch_get_key_info = key_info
        worker.m_store.batch_add_lease = lambda keys, ttl: [0] * len(keys)
        for req in metadata.requests:
            req.block_hashes = [f"{req.req_id}-{index}" for index in range(512)]
            req.block_ids_np = req.block_ids_by_group_np[0]
            req.save_keys = []
            req.load_spec.vllm_cached_tokens = 0
            # Remove fixture-generated locators: the native method must rebuild them.
            req.load_block_gvas_by_group_np = None
            req.partial_load_gva_per_group = []
        worker._prepare_load_gvas(metadata.requests)
        self.assertEqual(selected_req.load_block_gvas_by_group_np[0][512], 0)
        partial_key = worker._make_layerwise_partial_key(selected_req, 0, 512, 65537)
        partial_gva = infos[partial_key].gva_list()[0]
        self.assertEqual(selected_req.partial_load_gva_per_group, [partial_gva])
        self.assertEqual(worker._step_lease_result[partial_key], 0)
        self.assertIn(partial_key, selected._leased_infos)

        builder_type = load_methods(
            "kv_transfer.py",
            "LayerBatchBuilder",
            {
                "_ensure_buf",
                "_dedupe_transfer_blocks",
                "_require_request_arrays",
                "build_shared",
            },
            namespace,
        )
        builder = builder_type()
        builder.group_id = 0
        builder._block_ids_buf = builder._block_gvas_buf = None
        worker.layer_load_tasks = [[] for _ in range(worker.num_layers)]
        worker.layer_save_tasks = [[] for _ in range(worker.num_layers)]
        for layer in range(worker.num_layers):
            worker._process_load_for_layer_batch(metadata.requests, layer, 0, layer)
            task = worker.layer_load_tasks[layer][0]
            self.assertEqual(task.block_ranges[0].partial_block_index, 512)
            task.shared_block_data = builder.build_shared(task, is_save=False)
            shared = task.shared_block_data
            self.assertEqual(int(shared.block_gvas_arr[shared.block_ids_arr == 513][0]), partial_gva)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertIsNotNone(step)
        self.assertEqual([plan.selected for plan in step.requests], [True, False])
        for component in selected.components:
            pages = step.page_bases(component.layer_name)
            self.assertEqual(int(pages[513]), 0)
            self.assertTrue(np.all(pages[1:513] > 0))
            shared = worker.layer_load_tasks[component.layer_id][0].shared_block_data
            keep = shared.main_load_filter.keep_main
            self.assertTrue(keep[shared.block_ids_arr == 513].all())
            self.assertTrue(keep[shared.block_ids_arr >= 1000].all())
            self.assertFalse(keep[shared.block_ids_arr < 513].any())
        self.assertNotIn(partial_gva, exporter.calls[0][0])
        self.assertEqual(len(exporter.calls[0][0]), 512)
        finish_readers(step)
        self.assertTrue(step.finish(Event()))
        selected.close()


if __name__ == "__main__":
    unittest.main()
