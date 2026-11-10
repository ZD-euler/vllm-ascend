# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for native-layout selected reads; no SDK or NPU required."""

import ast
import ctypes
import dataclasses
import importlib.util
import json
import sys
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
STORE = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"
PACKAGE = "_selected_store_cpu_tests"
package = ModuleType(PACKAGE)
package.__path__ = [str(STORE)]
sys.modules.setdefault(PACKAGE, package)


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", STORE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_methods(filename, class_name, method_names, namespace):
    """Run actual native methods while replacing only external device imports."""
    tree = ast.parse((STORE / filename).read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    selected = ast.ClassDef(
        name=class_name,
        bases=[],
        keywords=[],
        decorator_list=[],
        body=[node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in method_names],
    )
    body = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(body), str(STORE / filename), "exec"), namespace)
    return namespace[class_name]


workspace_name = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_workspace"
workspace_spec = importlib.util.spec_from_file_location(
    workspace_name, ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/prefill_selected_workspace.py"
)
workspace = importlib.util.module_from_spec(workspace_spec)
sys.modules[workspace_name] = workspace
workspace_spec.loader.exec_module(workspace)
source = load_module("selected_kv_source")
address = sys.modules[f"{PACKAGE}.selected_kv_address"]


class Event:
    def __init__(self, done=True):
        self.done = done
        self.waits = 0
        self.recorded = False

    def record(self):
        self.recorded = True

    def query(self):
        return self.done

    def synchronize(self):
        self.waits += 1
        self.done = True


class Cache:
    def __init__(self, pointer, blocks):
        self.pointer = pointer
        self.shape = (blocks, 128, 1, 656)
        self.dtype = torch.int8
        self.device = torch.device("cpu")

    def data_ptr(self):
        return self.pointer

    def is_contiguous(self):
        return True


@dataclasses.dataclass
class Shared:
    block_ids_arr: np.ndarray
    block_gvas_arr: np.ndarray
    req_ids: list
    is_last_chunks: list
    load_keys: list
    main_load_filter: object = None
    block_keys: object = None
    save_keys: list = dataclasses.field(default_factory=list)


class Exporter:
    def __init__(self):
        self.calls = []

    def export_component_arrays(self, gvas, sizes, offset, page_bytes):
        self.calls.append((gvas.copy(), sizes.copy(), offset, page_bytes))
        return gvas + np.uint64(1 << 40)


def request(request_id="selected", h=65536, q=256, first=1, gva_first=None):
    count = (h + 127) // 128
    blocks = np.arange(first, first + (h + q + 127) // 128, dtype=np.int32)
    base = (1 << 24) + (first - 1) * (1 << 20) if gva_first is None else gva_first
    gvas = np.arange(count, dtype=np.uint64) * np.uint64(1 << 20) + np.uint64(base)
    partial = int(gvas[-1]) if h % 128 else 0
    if partial:
        gvas[-1] = 0  # Native stores the partial-page locator separately.
    return SimpleNamespace(
        req_id=request_id,
        target_token_len=h + q,
        # Test-only input for the separate SchedulerOutput dictionary.
        fixture_q=q,
        num_prompt_tokens=h + q,
        is_last_chunk=True,
        block_ids_by_group=[blocks.tolist()],
        block_ids_by_group_np=[blocks],
        load_block_gvas_by_group_np=[gvas],
        partial_load_gva_per_group=[partial],
        load_gva_block_offset=0,
        load_keys=[request_id],
        load_spec=SimpleNamespace(can_load=True, kvpool_cached_tokens=h, kvpool_store_skip_tokens=h),
    )


def fixture(requests=None, *, upload=lambda values, device: values.copy(), indexers=False):
    requests = requests or [request()]
    blocks, layers = 4096, 3
    names = [f"layer.{i}.main" for i in range(layers)]
    page_sizes = np.tile([128 * 656, 128 * 128, 128 * (2 if indexers else 4)], layers).astype(np.int64)
    addresses = np.arange(layers * 3, dtype=np.int64) * 100000000 + 1000000000
    caches = {name: (Cache(int(addresses[i * 3]), blocks),) for i, name in enumerate(names)}
    registry = {
        i: SimpleNamespace(main=SimpleNamespace(layer_name=name), extra_main_specs=()) for i, name in enumerate(names)
    }
    if indexers:
        for layer in range(layers):
            li_name = f"layer.{layer}.indexer"
            registry[layer].indexer = SimpleNamespace(layer_name=li_name)
            li_caches = []
            for component, dtype, elements in ((1, torch.int8, 16384), (2, torch.float16, 128)):
                cache = Cache(int(addresses[layer * 3 + component]), blocks)
                cache.dtype = dtype
                cache.shape = (blocks, elements)
                cache.numel = lambda n=blocks * elements: n
                cache.element_size = lambda size=(1 if dtype == torch.int8 else 2): size
                li_caches.append(cache)
            caches[li_name] = tuple(li_caches)
    builder = SimpleNamespace(
        _layer_cache_entry_offsets_np=np.arange(layers + 1) * 3,
        _kv_caches_base_addr_np=addresses,
        _block_len_np=page_sizes,
        _block_stride_np=page_sizes.copy(),
        layer_byte_offset=512,
    )
    removed = []
    object_size = int(512 + page_sizes.sum())
    infos = {}
    for req in requests:
        gvas = req.load_block_gvas_by_group_np[0]
        if req.partial_load_gva_per_group[0]:
            gvas = np.r_[gvas[:-1], np.uint64(req.partial_load_gva_per_group[0])]
        infos[req.req_id] = SimpleNamespace(
            gva_list=lambda values=gvas: values.tolist(),
            size=lambda: object_size,
        )
    recv = SimpleNamespace(group_builders=[builder], layer_batch_builder=builder, read_lease_release=None)
    worker = SimpleNamespace(
        use_layerwise_transfer=True,
        layerwise_offload=True,
        use_block_key_layerwise=False,
        load_async=False,
        num_kv_cache_groups=1,
        grouped_block_size=[128],
        pp_size=1,
        dcp_size=1,
        num_blocks=blocks,
        num_layers=layers,
        kv_caches=caches,
        kv_recv_thread=recv,
        selected_kv_source=None,
        _as_cache_tuple=lambda value: value if isinstance(value, tuple) else (value,),
        _layerwise_reuse_layout=SimpleNamespace(
            independent_layers=(),
            buffer_slots=((0, 2), (1,)),
            layer_cache_specs=registry,
        ),
        external_slot_release_waiter=None,
        next_layer_to_submit=0,
        m_store=SimpleNamespace(batch_remove_lease=lambda keys: removed.append(tuple(keys)) or 0),
        _invalid_block_ids_lock=threading.Lock(),
        _invalid_block_ids=set(),
        _step_keyinfo_cache=infos,
        _step_lease_result={key: 0 for key in infos},
    )

    def install(waiter):
        worker.external_slot_release_waiter = recv.external_slot_release_waiter = waiter
        return True

    worker.set_external_slot_release_waiter = install
    reset_native_tasks(worker, requests)
    exporter = Exporter()
    selected = source.SelectedKVSource(
        worker,
        exporter=exporter,
        event_factory=Event,
        upload=upload,
    )
    selected.begin_prepare()
    selected.record_leased_infos(list(infos), list(infos.values()))
    metadata = SimpleNamespace(requests=requests)
    selected.native_prepared(metadata)
    return (
        selected,
        worker,
        metadata,
        removed,
        exporter,
    )


def scheduled_tokens(metadata):
    return {req.req_id: req.fixture_q for req in metadata.requests}


def reset_native_tasks(worker, requests):
    """Use the same page IDs and GVA sources as the original request metadata."""
    locators = []
    for req in requests:
        gvas = req.load_block_gvas_by_group_np[0]
        if req.partial_load_gva_per_group[0]:
            gvas = np.r_[gvas[:-1], np.uint64(req.partial_load_gva_per_group[0])]
        locators.append(gvas)
    shared = Shared(
        np.concatenate([r.block_ids_by_group_np[0][: len(r.load_block_gvas_by_group_np[0])] for r in requests]),
        np.concatenate(locators),
        [r.req_id for r in requests],
        [True] * len(requests),
        [r.req_id for r in requests],
    )
    saved_ids = np.concatenate(
        [
            r.block_ids_by_group_np[0][len(gvas) - bool(r.partial_load_gva_per_group[0]) :]
            for r, gvas in zip(requests, locators)
        ]
    )
    saved = dataclasses.replace(shared, block_ids_arr=saved_ids, block_gvas_arr=saved_ids.astype(np.uint64) + (1 << 40))
    worker.layer_load_tasks = [[transfer_task(shared, i)] for i in range(worker.num_layers)]
    worker.layer_save_tasks = [[transfer_task(saved, i)] for i in range(worker.num_layers)]


def transfer_task(shared, layer_id):
    return SimpleNamespace(
        shared_block_data=shared, group_id=0, layer_idx_in_group=layer_id, use_key_major_ranges=False
    )


def finish_readers(step):
    for component in step._source.components:
        step.native_layer_ready(component.layer_id)


class ClassificationTest(unittest.TestCase):
    def test_actual_q_not_rounded_save_or_lookup(self):
        req = request()
        req.save_end_token = req.target_token_len - 127
        req.load_spec = SimpleNamespace(kvpool_cached_tokens=1)
        plan = source.classify_request(req, req.fixture_q)
        self.assertEqual((plan.history_tokens, plan.query_tokens, plan.selected), (65536, 256, True))

    def test_ineligible_and_unknown_chunks_remain_full(self):
        for changes in (
            {"fixture_q": None},
            {"fixture_q": 0},
            {"fixture_q": True},
            {"fixture_q": 2048},
            {"target_token_len": 16383 + 256},
        ):
            with self.subTest(changes=changes):
                req = request()
                for name, value in changes.items():
                    setattr(req, name, value)
                self.assertFalse(source.classify_request(req, req.fixture_q).selected)

    def test_selection_has_no_payload_arena_bound(self):
        self.assertTrue(source.classify_request(request(h=524288, q=1024), 1024).selected)

    def test_history_and_strict_query_ratio_boundaries(self):
        for h, q, expected in (
            (16383, 1, False),
            (16384, 249, True),
            (16384, 250, False),
            (19701, 300, True),
            (19700, 300, False),  # Q/(H+Q) is exactly 1.5%.
            (19699, 300, False),
            (20000, 300, True),  # Q/H would reject this request.
            (20480, 128, True),
            (20480, 311, True),
            (20480, 312, False),
            (65536, 998, True),
            (65536, 999, False),
            (524288, 7984, True),
            (524288, 7985, False),
            (524289, 128, False),
        ):
            with self.subTest(H=h, Q=q):
                self.assertIs(source.classify_request(request(h=h, q=q), q).selected, expected)
        self.assertEqual(source.MIN_HISTORY, workspace.MIN_HISTORY_TOKENS)

    def test_large_queries_are_selected_by_ratio_without_an_independent_cap(self):
        for q in (1025, 2048, 4096, 7984):
            with self.subTest(Q=q):
                self.assertLess(q / (524288 + q), 0.015)
                self.assertTrue(source.classify_request(request(h=524288, q=q), q).selected)

    def test_actual_unaligned_h_and_q_are_independent_of_request_phase(self):
        for h in (16384, 16385, 16511, 65536, 65537, 65663, 524287):
            for q in (1, 64, 125, 127, 129):
                with self.subTest(h=h, q=q):
                    req = request(h=h, q=q)
                    plan = source.classify_request(req, req.fixture_q)
                    self.assertEqual((plan.history_tokens, plan.query_tokens, plan.selected), (h, q, True))
                    req.num_prompt_tokens = h
                    req.is_last_chunk = False
                    self.assertTrue(source.classify_request(req, req.fixture_q).selected)
        for q in (1023, 1024):
            self.assertTrue(source.classify_request(request(h=131073, q=q), q).selected)

    def test_classification_does_not_read_request_phase_fields(self):
        req = request()
        del req.is_last_chunk, req.num_prompt_tokens
        self.assertTrue(source.classify_request(req, req.fixture_q).selected)


class SelectedSourceTest(unittest.TestCase):
    def setUp(self):
        self.caller_stream = object()
        self.current_stream = self.caller_stream
        self.stream_entries = []

        @contextmanager
        def stream_context(stream):
            previous = self.current_stream
            self.stream_entries.append(stream)
            self.current_stream = stream
            try:
                yield
            finally:
                self.current_stream = previous

        self.npu = SimpleNamespace(
            Stream=Mock(side_effect=lambda device: SimpleNamespace(device=device)),
            stream=stream_context,
        )
        patcher = patch.object(torch, "npu", self.npu, create=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_nonfinal_and_post_prompt_steps_keep_source_coverage_checks(self):
        for q, prompt_tokens in ((2048, 600000), (1, 262144)):
            for complete in (True, False):
                with self.subTest(Q=q, prompt_tokens=prompt_tokens, complete=complete):
                    req = request(h=262144, q=q)
                    req.is_last_chunk = False
                    req.num_prompt_tokens = prompt_tokens
                    selected, worker, metadata, _, exporter = fixture([req])
                    if not complete:
                        req.load_spec.kvpool_cached_tokens = 0
                        req.load_spec.kvpool_store_skip_tokens = 0
                    step = selected.prepare(metadata, scheduled_tokens(metadata))
                    if complete:
                        self.assertIsNotNone(step)
                        self.assertTrue(step.requests[0].selected)
                        finish_readers(step)
                        self.assertTrue(step.finish(Event()))
                    else:
                        self.assertIsNone(step)
                        self.assertEqual(exporter.calls, [])
                        for tasks in worker.layer_load_tasks:
                            self.assertIsNone(tasks[0].shared_block_data.main_load_filter)
                    selected.close()

    def test_partial_history_page_keeps_native_load_and_zero_selected_address(self):
        for h, q in ((65537, 1), (65537, 125), (65663, 127), (65663, 983)):
            with self.subTest(h=h, q=q):
                req = request(h=h, q=q)
                selected, worker, metadata, _, exporter = fixture([req])
                # Even without a save task, the H/Q boundary must remain FULL.
                worker.layer_save_tasks = [[] for _ in range(worker.num_layers)]
                step = selected.prepare(metadata, scheduled_tokens(metadata))
                self.assertIsNotNone(step)
                self.assertTrue(step.requests[0].selected)
                full_pages = h // 128
                tail = req.block_ids_by_group_np[0][full_pages]
                for component in selected.components:
                    pages = step.page_bases(component.layer_name)
                    self.assertEqual(int(pages[tail]), 0)
                    self.assertTrue(np.all(pages[req.block_ids_by_group_np[0][:full_pages]] > 0))
                    shared = worker.layer_load_tasks[component.layer_id][0].shared_block_data
                    self.assertTrue(shared.main_load_filter.keep_main[-1])
                    self.assertFalse(shared.main_load_filter.keep_main[:-1].any())
                self.assertEqual(len(exporter.calls[0][0]), full_pages)
                finish_readers(step)
                self.assertTrue(step.finish(Event()))
                selected.close()

    def test_partial_history_requires_real_tail_lease_and_each_layer_load(self):
        for missing in ("locator", "lease", "layer"):
            with self.subTest(missing=missing):
                req = request(h=65537, q=127)
                selected, worker, metadata, _, exporter = fixture([req])
                if missing == "locator":
                    req.partial_load_gva_per_group = []
                elif missing == "lease":
                    selected._leased_infos[req.req_id] = SimpleNamespace(
                        gva_list=lambda req=req: req.load_block_gvas_by_group_np[0].tolist(),
                        size=lambda: 1 << 24,
                    )
                else:
                    shared = worker.layer_load_tasks[-1][0].shared_block_data
                    worker.layer_load_tasks[-1][0].shared_block_data = dataclasses.replace(
                        shared, block_ids_arr=shared.block_ids_arr[:-1], block_gvas_arr=shared.block_gvas_arr[:-1]
                    )
                self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
                self.assertEqual(exporter.calls, [])
                for tasks in worker.layer_load_tasks:
                    self.assertIsNone(tasks[0].shared_block_data.main_load_filter)
                selected.close()

    def test_address_buffers_allocate_once_before_hooks_and_publish_reuses_them(self):
        injected, _, _, _, _ = fixture()
        old_source, worker, _, _, exporter = fixture()
        old_source.close()
        allocations, allocation_streams, copies = [], [], []
        real_empty, real_copy = torch.empty, torch.Tensor.copy_

        def empty(*args, **kwargs):
            self.assertIsNone(worker.external_slot_release_waiter)
            self.assertIsNone(worker.kv_recv_thread.read_lease_release)
            allocations.append((args, kwargs.copy()))
            allocation_streams.append(self.current_stream)
            kwargs.pop("pin_memory", None)  # Device allocator leaf only.
            return real_empty(*args, **kwargs)

        def copy(target, host, *args, **kwargs):
            copies.append((target.data_ptr(), host.data_ptr(), tuple(host.shape), kwargs))
            return real_copy(target, host, *args, **kwargs)

        with patch.object(source.torch, "empty", side_effect=empty), patch.object(torch.Tensor, "copy_", copy):
            selected = source.SelectedKVSource(worker, exporter=exporter, event_factory=Event)
            capacity = worker.num_blocks * worker.num_layers
            self.assertEqual(len(allocations), 2)
            self.assertEqual([args for args, _ in allocations], [(capacity,), (capacity,)])
            self.assertTrue(allocations[0][1]["pin_memory"])
            self.assertEqual(allocations[1][1]["device"], worker.kv_caches["layer.0.main"][0].device)
            self.npu.Stream.assert_called_once_with(device=selected.device)
            self.assertEqual(allocation_streams, [self.caller_stream, selected._metadata_stream])
            self.assertIs(self.current_stream, self.caller_stream)
            host, device = selected._metadata_buffer
            for index, pages in enumerate(([1, 7], [4, 2], [3, 9])):
                ids = np.array(pages, dtype=np.int32)
                refs = np.array([(1 << 63) + index * 10000 + 19, (1 << 63) + index * 10000 + 43], dtype=np.uint64)
                expected = np.zeros((worker.num_layers, max(pages) + 1), dtype=np.uint64)
                for component in selected.components:
                    for page, ref in zip(pages, refs):
                        expected[component.layer_id, page] = int(ref) + component.object_offset
                with patch.object(source.np, "copyto", side_effect=AssertionError("unexpected dense CPU copy")):
                    published = selected._publish(ids, refs, max(pages) + 1)
                    uploaded = injected._publish(ids, refs, max(pages) + 1)
                np.testing.assert_array_equal(published.numpy(), expected)
                np.testing.assert_array_equal(uploaded, expected)
                np.testing.assert_array_equal(host[: expected.size].view(expected.shape).numpy(), expected)
                self.assertEqual(published.data_ptr(), device.data_ptr())
                self.assertEqual(len(copies), index + 1)
                self.assertEqual(
                    copies[-1], (device.data_ptr(), host.data_ptr(), expected.shape, {"non_blocking": True})
                )
            # Reject an oversized domain before clearing the prior view or enqueueing a copy.
            for publisher in (selected, injected):
                with self.assertRaisesRegex(ValueError, "startup byte reservation"):
                    publisher._publish(np.array([worker.num_blocks], dtype=np.int32), refs[:1], worker.num_blocks + 1)
            np.testing.assert_array_equal(host[: expected.size].view(expected.shape).numpy(), expected)
            self.assertEqual(len(copies), 3)
            self.assertEqual(len(allocations), 2)
        selected.close()
        injected.close()

    def test_default_upload_and_ready_use_the_source_stream(self):
        real_empty, real_copy = torch.empty, torch.Tensor.copy_
        operations = []

        def empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        def copy(target, host, *args, **kwargs):
            operations.append(("copy", self.current_stream))
            return real_copy(target, host, *args, **kwargs)

        def current():
            return self.current_stream

        class Ready(Event):
            def record(self):
                operations.append(("ready", current()))
                super().record()

        with patch.object(source.torch, "empty", side_effect=empty):
            selected, worker, metadata, removed, _ = fixture(upload=None)
        selected._event = Ready
        upload_stream = selected._metadata_stream
        with patch.object(torch.Tensor, "copy_", copy):
            step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertEqual(operations, [("copy", upload_stream), ("ready", upload_stream)])
        self.assertIs(self.current_stream, self.caller_stream)
        self.assertIsNot(upload_stream, self.caller_stream)
        self.assertEqual(upload_stream.device, selected.device)
        self.assertEqual(self.npu.Stream.call_count, 1)
        self.assertEqual(step.ready("layer.0.main").waits, 0)
        self.assertEqual(removed, [])
        self.assertIsNotNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)
        finish_readers(step)
        self.assertTrue(step.finish(Event()))
        selected.close()
        self.assertIsNone(selected._metadata_stream)

    def test_native_fallback_does_not_submit_address_metadata(self):
        real_empty = torch.empty

        def empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        with patch.object(source.torch, "empty", side_effect=empty):
            selected, worker, metadata, _, _ = fixture([request(q=4096)], upload=None)
        entries = len(self.stream_entries)
        selected._event = Mock(side_effect=AssertionError("native fallback recorded selected ready"))
        with patch.object(selected, "_publish", side_effect=AssertionError("native fallback uploaded addresses")):
            self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
        self.assertEqual(len(self.stream_entries), entries)
        self.assertIs(self.current_stream, self.caller_stream)
        for tasks in worker.layer_load_tasks:
            self.assertIsNone(tasks[0].shared_block_data.main_load_filter)
        selected.close()

    def test_live_and_failed_source_cannot_overwrite_production_address_buffers(self):
        real_empty = torch.empty

        def empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        with patch.object(source.torch, "empty", side_effect=empty):
            selected, _, metadata, removed, _ = fixture(upload=None)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        host, device = selected._metadata_buffer
        before_host, before_device = host.clone(), device.clone()
        for failed in (False, True):
            if failed:
                step.finish(failed=True)
            with patch.object(selected, "_publish", side_effect=AssertionError("live storage overwritten")):
                with self.assertRaisesRegex(RuntimeError, "ownership retained"):
                    selected.begin_prepare()
                with self.assertRaisesRegex(RuntimeError, "before native load"):
                    selected.prepare(metadata, scheduled_tokens(metadata))
            np.testing.assert_array_equal(host.numpy(), before_host.numpy())
            np.testing.assert_array_equal(device.numpy(), before_device.numpy())
            self.assertIs(selected.step, step)
            self.assertEqual(removed, [])

    def test_invalid_shared_metadata_or_mapping_fails_before_publish_and_task_updates(self):
        for fault in ("invalid_dtype", "conflicting_mapping", "wrong_layer"):
            with self.subTest(fault=fault):
                requests = [request()]
                if fault == "conflicting_mapping":
                    requests.append(request("other", gva_first=1 << 25))
                selected, worker, metadata, removed, _ = fixture(requests)
                if fault == "invalid_dtype":
                    shared = worker.layer_load_tasks[1][0].shared_block_data
                    worker.layer_load_tasks[1][0].shared_block_data = dataclasses.replace(
                        shared, block_ids_arr=shared.block_ids_arr.astype(object)
                    )
                elif fault == "wrong_layer":
                    worker.layer_load_tasks[1][0].layer_idx_in_group = 0
                tasks = tuple(tasks[0] for tasks in worker.layer_load_tasks)
                before = tuple(task.shared_block_data for task in tasks)
                with patch.object(selected, "_publish", wraps=selected._publish) as publish:
                    with self.assertRaises(ValueError):
                        selected.prepare(metadata, scheduled_tokens(metadata))
                    publish.assert_not_called()
                for task, shared in zip(tasks, before):
                    self.assertIs(task.shared_block_data, shared)
                    self.assertIsNone(shared.main_load_filter)
                self.assertIsNone(selected.step)
                self.assertFalse(selected._publication_failed)
                self.assertFalse(hasattr(metadata, "selected_kv_step"))
                self.assertEqual(removed, [])
                selected.close()

    def test_publication_failure_retains_buffers_and_leases_without_retry(self):
        for fault in ("copy", "ready"):
            with self.subTest(fault=fault):
                real_empty, real_copy = torch.empty, torch.Tensor.copy_

                def empty(*args, _real_empty=real_empty, **kwargs):
                    kwargs.pop("pin_memory", None)
                    return _real_empty(*args, **kwargs)

                with patch.object(source.torch, "empty", side_effect=empty):
                    selected, worker, metadata, removed, _ = fixture(upload=None)
                failure = RuntimeError("publication failure")

                def copy(target, value, *args, _real_copy=real_copy, _fault=fault, _failure=failure, **kwargs):
                    _real_copy(target, value, *args, **kwargs)
                    if _fault == "copy":
                        raise _failure
                    return target

                class FailedReady(Event):
                    def record(self, _failure=failure):
                        raise _failure

                if fault == "ready":
                    selected._event = FailedReady
                backing = selected._metadata_buffer
                upload_stream = selected._metadata_stream
                old_shared = worker.layer_load_tasks[0][0].shared_block_data
                with patch.object(torch.Tensor, "copy_", copy), self.assertRaises(RuntimeError) as caught:
                    selected.prepare(metadata, scheduled_tokens(metadata))
                self.assertIs(caught.exception, failure)
                self.assertTrue(selected._publication_failed)
                self.assertIs(selected._metadata_buffer, backing)
                self.assertIs(selected._metadata_stream, upload_stream)
                self.assertIs(self.current_stream, self.caller_stream)
                self.assertFalse(hasattr(metadata, "selected_kv_step"))
                self.assertIs(worker.layer_load_tasks[0][0].shared_block_data, old_shared)
                for operation in (
                    selected.begin_prepare,
                    selected.close,
                    lambda selected=selected, metadata=metadata: selected.prepare(metadata, scheduled_tokens(metadata)),
                    lambda selected=selected: selected._release_native(["selected"]),
                ):
                    with self.assertRaisesRegex(RuntimeError, "ownership retained"):
                        operation()
                self.assertEqual(removed, [])
                self.assertIs(selected._metadata_buffer, backing)
                self.assertIs(selected._metadata_stream, upload_stream)

    def test_startup_stream_failure_never_installs_source_hooks(self):
        old_source, worker, _, _, exporter = fixture()
        old_source.close()
        real_empty = torch.empty

        def empty(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        with (
            patch.object(source.torch, "empty", side_effect=empty),
            patch.object(self.npu, "Stream", side_effect=RuntimeError("stream allocation failed")),
            self.assertRaisesRegex(RuntimeError, "stream allocation failed"),
        ):
            source.SelectedKVSource(worker, exporter=exporter, event_factory=Event)
        self.assertIsNone(worker.kv_recv_thread.read_lease_release)
        self.assertIs(self.current_stream, self.caller_stream)

    def test_startup_allocation_failure_never_installs_source_hooks(self):
        for fail_at in (1, 2):
            with self.subTest(fail_at=fail_at):
                old_source, worker, _, _, exporter = fixture()
                old_source.close()
                calls = []
                real_empty = torch.empty

                def empty(*args, _calls=calls, _fail_at=fail_at, _real_empty=real_empty, **kwargs):
                    _calls.append(True)
                    if len(_calls) == _fail_at:
                        raise MemoryError("injected startup allocation failure")
                    kwargs.pop("pin_memory", None)
                    return _real_empty(*args, **kwargs)

                with patch.object(source.torch, "empty", side_effect=empty), self.assertRaises(MemoryError):
                    source.SelectedKVSource(worker, exporter=exporter, event_factory=Event)
                self.assertIsNone(worker.external_slot_release_waiter)
                self.assertIsNone(worker.kv_recv_thread.read_lease_release)

    def test_injected_upload_does_not_allocate_production_buffers(self):
        with patch.object(source.torch, "empty", side_effect=AssertionError("unexpected allocation")):
            selected, _, _, _, _ = fixture()
            self.assertIsNone(selected._metadata_buffer)
            self.assertIsNone(selected._metadata_stream)
            self.npu.Stream.assert_not_called()
            selected.close()

    def test_unsupported_cache_dtype_is_rejected_before_installing_hooks(self):
        selected, worker, _, _, exporter = fixture()
        selected.close()
        next(iter(worker.kv_caches.values()))[0].dtype = torch.uint8
        with self.assertRaisesRegex(ValueError, "packed C8 cache view"):
            source.SelectedKVSource(worker, exporter=exporter)
        self.assertIsNone(worker.external_slot_release_waiter)
        self.assertIsNone(worker.kv_recv_thread.read_lease_release)

    def test_only_post_lease_address_and_extent_can_authorize_selected_reads(self):
        selected, worker, metadata, _, exporter = fixture()
        req = metadata.requests[0]
        old_info = worker._step_keyinfo_cache[req.req_id]
        moved = req.load_block_gvas_by_group_np[0] + np.uint64(1 << 32)
        req.load_block_gvas_by_group_np[0] = moved
        selected.record_leased_infos(
            [req.req_id], [SimpleNamespace(gva_list=lambda: moved.tolist(), size=old_info.size)]
        )
        reset_native_tasks(worker, metadata.requests)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertIsNotNone(step)
        np.testing.assert_array_equal(exporter.calls[0][0], moved)
        self.assertIs(worker._step_keyinfo_cache[req.req_id], old_info)

        selected2, worker2, metadata2, _, exporter2 = fixture()
        req2 = metadata2.requests[0]
        selected2.record_leased_infos(
            [req2.req_id],
            [SimpleNamespace(gva_list=lambda: req2.load_block_gvas_by_group_np[0].tolist(), size=lambda: 1)],
        )
        self.assertIsNone(selected2.prepare(metadata2, scheduled_tokens(metadata2)))
        self.assertEqual(exporter2.calls, [])
        self.assertGreater(worker2._step_keyinfo_cache[req2.req_id].size(), 1)

    def test_new_prepare_cannot_reuse_stale_leased_descriptor(self):
        selected, worker, metadata, _, exporter = fixture()
        selected.begin_prepare()
        reset_native_tasks(worker, metadata.requests)
        selected.native_prepared(metadata)
        self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
        self.assertEqual(exporter.calls, [])
        self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)

    def test_different_blob_extents_export_needed_span_in_one_batch(self):
        requests = [request("short"), request("long", first=1025)]
        selected, _, metadata, _, exporter = fixture(requests)
        required = max(c.object_offset + c.page_bytes for c in selected.components)
        sizes = (required, required + 8192)
        selected.record_leased_infos(
            [r.req_id for r in requests],
            [
                SimpleNamespace(gva_list=lambda r=r: r.load_block_gvas_by_group_np[0], size=lambda size=size: size)
                for r, size in zip(requests, sizes)
            ],
        )

        step = selected.prepare(metadata, scheduled_tokens(metadata))

        self.assertIsNotNone(step)
        self.assertEqual(len(exporter.calls), 1)
        gvas, extents, offset, length = exporter.calls[0]
        np.testing.assert_array_equal(gvas, np.concatenate([r.load_block_gvas_by_group_np[0] for r in requests]))
        np.testing.assert_array_equal(extents, np.repeat(sizes, 512))
        self.assertEqual((offset, length), (0, required))
        for component in selected.components:
            for req in requests:
                ids = req.block_ids_by_group_np[0][:512]
                expected = req.load_block_gvas_by_group_np[0] + np.uint64((1 << 40) + component.object_offset)
                np.testing.assert_array_equal(step.page_bases(component.layer_name)[ids], expected)

    def test_object_short_of_last_main_byte_falls_back_before_export(self):
        selected, worker, metadata, _, exporter = fixture()
        req = metadata.requests[0]
        required = max(c.object_offset + c.page_bytes for c in selected.components)
        selected.record_leased_infos(
            [req.req_id],
            [SimpleNamespace(gva_list=lambda: req.load_block_gvas_by_group_np[0], size=lambda: required - 1)],
        )
        original = [tasks[0].shared_block_data for tasks in worker.layer_load_tasks]
        with patch.object(selected, "_publish", wraps=selected._publish) as publish:
            self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
            publish.assert_not_called()
        self.assertEqual(exporter.calls, [])
        for tasks, shared in zip(worker.layer_load_tasks, original):
            self.assertIs(tasks[0].shared_block_data, shared)
            self.assertIsNone(shared.main_load_filter)

    def test_native_component_identity_preserves_original_page_ids(self):
        selected, worker, metadata, _, exporter = fixture()
        req = metadata.requests[0]
        original_ids = req.block_ids_by_group_np[0].copy()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertIs(step, selected.step)
        self.assertIs(step.connector_metadata, metadata)
        np.testing.assert_array_equal(req.block_ids_by_group_np[0], original_ids)
        self.assertEqual(req.block_ids_by_group[0], original_ids.tolist())
        self.assertEqual(len(exporter.calls), 1)
        for component in selected.components:
            pages = step.page_bases(component.layer_name)
            expected = req.load_block_gvas_by_group_np[0] + np.uint64((1 << 40) + component.object_offset)
            np.testing.assert_array_equal(pages[1:513], expected)
            self.assertTrue(step.ready(component.layer_name).recorded)
        self.assertFalse(hasattr(selected.components[2], "previous_owner"))
        self.assertIsNone(worker.external_slot_release_waiter)

    def test_preparation_keeps_scheduler_payload_data_only_and_unchanged(self):
        def encode(value):
            if isinstance(value, SimpleNamespace):
                return vars(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            raise TypeError(f"Unexpected worker state in metadata: {type(value).__name__}")

        selected, _, metadata, _, _ = fixture()
        payload = json.dumps(metadata, default=encode, sort_keys=True)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertIs(step.connector_metadata, metadata)
        self.assertEqual(json.dumps(metadata, default=encode, sort_keys=True), payload)
        self.assertFalse(hasattr(metadata, "selected_kv_step"))

    def test_selected_native_selected_steps_publish_only_the_current_source(self):
        selected, worker, first_metadata, removed, _ = fixture()
        first = selected.prepare(first_metadata, scheduled_tokens(first_metadata))
        finish_readers(first)
        first.finish(completion=Event(False))
        selected.begin_prepare()
        self.assertIsNone(selected.step)
        self.assertEqual(removed, [("selected",)])

        native_metadata = SimpleNamespace(requests=[request(q=2048)])
        reset_native_tasks(worker, native_metadata.requests)
        selected.native_prepared(native_metadata)
        self.assertIsNone(selected.prepare(native_metadata, scheduled_tokens(native_metadata)))
        self.assertIsNone(selected.step)

        selected.begin_prepare()
        next_metadata = SimpleNamespace(requests=first_metadata.requests)
        reset_native_tasks(worker, next_metadata.requests)
        infos = worker._step_keyinfo_cache
        selected.record_leased_infos(list(infos), list(infos.values()))
        selected.native_prepared(next_metadata)
        second = selected.prepare(next_metadata, scheduled_tokens(next_metadata))
        self.assertIsNot(second, first)
        self.assertIs(selected.step, second)
        self.assertIs(second.connector_metadata, next_metadata)
        finish_readers(second)
        second.finish(completion=Event())
        selected.close()
        self.assertEqual(removed, [("selected",), ("selected",)])
        self.assertIsNone(worker.kv_recv_thread.read_lease_release)

    def test_empty_step_does_not_publish_or_reuse_the_previous_selection(self):
        selected, worker, metadata, removed, exporter = fixture()
        first = selected.prepare(metadata, scheduled_tokens(metadata))
        finish_readers(first)
        first.finish(completion=Event(False))
        selected.begin_prepare()
        worker.layer_load_tasks = [[] for _ in selected.components]
        worker.layer_save_tasks = [[] for _ in selected.components]
        empty_metadata = SimpleNamespace(requests=[])
        selected.native_prepared(empty_metadata)
        self.assertIsNone(selected.prepare(empty_metadata, {}))
        self.assertIsNone(selected.step)
        self.assertEqual(len(exporter.calls), 1)
        self.assertEqual(removed, [("selected",)])

        selected.begin_prepare()
        next_metadata = SimpleNamespace(requests=metadata.requests)
        reset_native_tasks(worker, next_metadata.requests)
        infos = worker._step_keyinfo_cache
        selected.record_leased_infos(list(infos), list(infos.values()))
        selected.native_prepared(next_metadata)
        current = selected.prepare(next_metadata, scheduled_tokens(next_metadata))
        self.assertIsNot(current, first)
        self.assertIs(current.connector_metadata, next_metadata)
        finish_readers(current)
        current.finish(completion=Event())
        selected.close()
        self.assertEqual(removed, [("selected",), ("selected",)])
        self.assertIsNone(worker.kv_recv_thread.read_lease_release)

    def test_leased_addresses_follow_nonsequential_native_page_ids(self):
        req = request()
        original_ids = np.roll(req.block_ids_by_group_np[0], 7)
        req.block_ids_by_group_np = [original_ids.copy()]
        req.block_ids_by_group = [original_ids.tolist()]
        selected, worker, metadata, _, _ = fixture([req])
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        history_ids, query_ids = original_ids[:512], original_ids[512:]
        for component in selected.components:
            pages = step.page_bases(component.layer_name)
            expected = req.load_block_gvas_by_group_np[0] + np.uint64((1 << 40) + component.object_offset)
            np.testing.assert_array_equal(pages[history_ids], expected)
            np.testing.assert_array_equal(pages[query_ids], 0)
        shared = worker.layer_load_tasks[0][0].shared_block_data
        np.testing.assert_array_equal(
            shared.main_load_filter.keep_main,
            ~np.isin(shared.block_ids_arr, history_ids),
        )
        np.testing.assert_array_equal(req.block_ids_by_group_np[0], original_ids)

    def test_shared_prefix_full_request_keeps_its_main_and_all_li(self):
        selected, worker, metadata, _, _ = fixture([request(), request("full", q=2048, first=10)])
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertEqual([item.selected for item in step.requests], [True, False])
        shared = worker.layer_load_tasks[0][0].shared_block_data
        filt = shared.main_load_filter
        rows = shared.block_ids_arr.size
        original = np.arange(rows * 3).reshape(rows, 3)
        addresses, sizes, gvas = filt.apply(0, original.ravel(), (original + 10000).ravel(), (original + 20000).ravel())
        expected = np.ones(original.shape, dtype=bool)
        expected[:9, 0] = False
        np.testing.assert_array_equal(addresses, original[expected])
        np.testing.assert_array_equal(sizes, (original + 10000)[expected])
        np.testing.assert_array_equal(gvas, (original + 20000)[expected])
        self.assertEqual(int(expected[:, 1:].sum()), rows * 2)

    def test_main_filter_preserves_order_duplicates_and_readonly_shared_mask(self):
        for dtype in (np.int32, np.int64):
            with self.subTest(dtype=dtype):
                selected, worker, metadata, _, _ = fixture([request(), request("full", q=2048, first=10)])
                shared = worker.layer_load_tasks[0][0].shared_block_data
                order = np.r_[np.arange(shared.block_ids_arr.size)[::-1], [0, 8, 0, 1]]
                shared.block_ids_arr = shared.block_ids_arr[order].astype(dtype)
                shared.block_gvas_arr = shared.block_gvas_arr[order]
                selected.prepare(metadata, scheduled_tokens(metadata))
                filtered = worker.layer_load_tasks[0][0].shared_block_data
                keep = filtered.main_load_filter.keep_main
                np.testing.assert_array_equal(keep, ~np.isin(shared.block_ids_arr, np.arange(1, 10)))
                self.assertFalse(keep.flags.writeable)
                with self.assertRaises(ValueError):
                    keep[0] = False
                self.assertIs(filtered.block_ids_arr, shared.block_ids_arr)
                self.assertTrue(all(tasks[0].shared_block_data is filtered for tasks in worker.layer_load_tasks))
                addresses = np.arange(keep.size * 3, dtype=np.int64).reshape(-1, 3)
                mask = np.ones(addresses.shape, dtype=bool)
                mask[:, 0] = keep
                actual = filtered.main_load_filter.apply(0, addresses.ravel(), addresses.ravel(), addresses.ravel())
                for values in actual:
                    np.testing.assert_array_equal(values, addresses[mask])

    def test_current_q_rows_never_suppressed(self):
        selected, worker, metadata, _, _ = fixture()
        request = metadata.requests[0]
        query_pages = request.block_ids_by_group_np[0][512:]
        original = worker.layer_load_tasks[0][0].shared_block_data
        # Include actual Q pages in the native task: an empty slice proves nothing.
        shared = dataclasses.replace(
            original,
            block_ids_arr=np.r_[original.block_ids_arr, query_pages],
            block_gvas_arr=np.r_[original.block_gvas_arr, query_pages.astype(np.uint64) + (1 << 40)],
        )
        worker.layer_load_tasks = [[transfer_task(shared, i)] for i in range(worker.num_layers)]
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertIsNotNone(step)
        filtered = worker.layer_load_tasks[0][0].shared_block_data
        is_query = np.isin(filtered.block_ids_arr, query_pages)
        self.assertEqual(int(is_query.sum()), 2)
        self.assertTrue(filtered.main_load_filter.keep_main[is_query].all())
        self.assertFalse(filtered.main_load_filter.keep_main[~is_query].any())
        self.assertFalse(np.isin(query_pages, np.flatnonzero(step.page_bases("layer.0.main"))).any())
        finish_readers(step)
        self.assertTrue(step.finish(Event()))
        selected.close()

    def test_native_only_is_noop(self):
        selected, worker, metadata, removed, exporter = fixture([request(q=2048)])
        old_shared = worker.layer_load_tasks[0][0].shared_block_data
        self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
        self.assertIs(worker.layer_load_tasks[0][0].shared_block_data, old_shared)
        self.assertEqual(exporter.calls, [])
        worker.kv_recv_thread.read_lease_release(["selected"])
        self.assertEqual(removed, [("selected",)])

    def test_missing_ddr_coverage_falls_back_before_filter(self):
        for fault in ("missing_lease", "short_gvas", "load_offset", "native_error", "partial_ddr_hit"):
            with self.subTest(fault=fault):
                selected, worker, metadata, _, exporter = fixture()
                req = metadata.requests[0]
                if fault == "missing_lease":
                    worker._step_lease_result[req.req_id] = 7
                elif fault == "short_gvas":
                    req.load_block_gvas_by_group_np[0] = req.load_block_gvas_by_group_np[0][1:]
                elif fault == "load_offset":
                    req.load_gva_block_offset = 1
                elif fault == "partial_ddr_hit":
                    req.load_spec.kvpool_store_skip_tokens = 32768
                else:
                    worker._invalid_block_ids.add(1)
                self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
                self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)
                self.assertEqual(exporter.calls, [])

    def test_invalid_native_page_is_an_error_before_publish(self):
        for invalid in (0, -1, 4096):
            selected, worker, metadata, _, exporter = fixture()
            metadata.requests[0].block_ids_by_group_np[0][-1] = invalid
            with self.assertRaisesRegex(ValueError, "native page ID"):
                selected.prepare(metadata, scheduled_tokens(metadata))
            self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)
            self.assertEqual(exporter.calls, [])

    def test_conflicting_shared_page_rejected_without_partial_filter(self):
        selected, worker, metadata, _, _ = fixture([request(), request("other", gva_first=1 << 25)])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertTrue(
            all(task.shared_block_data.main_load_filter is None for tasks in worker.layer_load_tasks for task in tasks)
        )

    def test_mixed_unknown_q_falls_back_before_any_address_publish_or_filter(self):
        req = request("async", q=2048, first=10)
        req.fixture_q = None
        selected, worker, metadata, _, exporter = fixture([request(), req])
        self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
        self.assertEqual(exporter.calls, [])
        self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)

    def test_worker_rejects_missing_extra_zero_and_invalid_scheduled_requests_before_filter(self):
        for counts in (
            {},
            {"selected": 256, "extra": 128},
            {"selected": 0},
            {"selected": -1},
            {"selected": True},
            {"selected": None},
        ):
            with self.subTest(counts=counts):
                selected, worker, metadata, _, exporter = fixture()
                self.assertIsNone(selected.prepare(metadata, counts))
                self.assertEqual(exporter.calls, [])
                self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)

    def test_worker_reads_current_q_without_mutating_native_metadata_or_schedule(self):
        from types import MappingProxyType

        selected, worker, metadata, _, exporter = fixture()
        counts = MappingProxyType({"selected": 2048, "not_scheduled": 0})
        before = vars(metadata.requests[0]).copy()
        self.assertIsNone(selected.prepare(metadata, counts))
        self.assertEqual(vars(metadata.requests[0]), before)
        self.assertEqual(dict(counts), {"selected": 2048, "not_scheduled": 0})
        self.assertEqual(exporter.calls, [])
        self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)

    def test_unprepared_same_content_metadata_cannot_reuse_native_tasks(self):
        selected, worker, metadata, _, exporter = fixture()
        other = SimpleNamespace(requests=metadata.requests)
        with self.assertRaisesRegex(RuntimeError, "native preparation of this connector metadata"):
            selected.prepare(other, scheduled_tokens(other))
        self.assertEqual(exporter.calls, [])
        self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)
        self.assertIs(selected._prepared_metadata, metadata)

    def test_native_fallback_consumes_preparation_and_cannot_be_retried_with_changed_q(self):
        selected, worker, metadata, _, exporter = fixture()
        self.assertIsNone(selected.prepare(metadata, {"selected": 2048}))
        with self.assertRaisesRegex(RuntimeError, "native preparation of this connector metadata"):
            selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertEqual(exporter.calls, [])
        self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)

    def test_duplicate_request_ids_cannot_authorize_main_filter(self):
        for counts, duplicate_error in (
            ({"selected": None}, False),
            ({"selected": 256, "extra": 128}, False),
            ({"selected": 256}, True),
        ):
            with self.subTest(counts=counts):
                selected, worker, metadata, _, exporter = fixture()
                metadata.requests.append(metadata.requests[0])
                if duplicate_error:
                    with self.assertRaisesRegex(ValueError, "Duplicate request IDs"):
                        selected.prepare(metadata, counts)
                else:
                    self.assertIsNone(selected.prepare(metadata, counts))
                self.assertEqual(exporter.calls, [])
                self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)

    def test_native_load_submission_prevents_late_selection(self):
        selected, worker, metadata, _, exporter = fixture()
        worker.next_layer_to_submit = 1
        with self.assertRaisesRegex(RuntimeError, "before native load submission"):
            selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertEqual(exporter.calls, [])
        self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)

    def test_final_empty_native_load_and_selected_events_both_gate_retirement(self):
        selected, _, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        delayed = Event(False)
        finish_readers(step)
        step.finish(completion=delayed)
        self.assertFalse(step.try_retire())
        self.assertEqual(removed, [])
        selected.retire_previous()
        self.assertEqual(delayed.waits, 1)
        self.assertEqual(removed, [("selected",)])
        self.assertIsNone(selected.step)
        selected.close()
        self.assertIsNone(selected.worker.external_slot_release_waiter)
        self.assertIsNone(selected.worker.kv_recv_thread.read_lease_release)

    def test_native_callback_alone_does_not_release_selected_source(self):
        selected, worker, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        worker.kv_recv_thread.read_lease_release(["selected"])
        self.assertEqual(removed, [])
        self.assertFalse(step.try_retire())
        finish_readers(step)
        step.finish(completion=Event())
        self.assertEqual(removed, [("selected",)])

    def test_source_and_runtime_retirement_wait_for_completion_only_once(self):
        selected, _, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        completion = Event(False)
        finish_readers(step)
        step.finish(completion)
        runtime_class = load_methods(
            "../../sparse_kv_offload/prefill_selected_runtime.py",
            "SelectedKVRuntime",
            {"_retire_previous"},
            {},
        )
        runtime = runtime_class()
        runtime.source_step, runtime._failed = step, False
        runtime.selection_planner = SimpleNamespace(release_quiescent=Mock())
        runtime._metadata = ()
        selected.begin_prepare()
        self.assertTrue(step._closed)
        self.assertEqual(completion.waits, 1)
        runtime._retire_previous()
        self.assertEqual(completion.waits, 1)
        self.assertEqual(removed, [("selected",)])
        runtime.selection_planner.release_quiescent.assert_called_once_with()
        self.assertIsNone(runtime.source_step)
        selected.close()

    def test_failure_is_fast_fail_closed_and_never_reuses_storage(self):
        selected, _, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        step.finish(failed=True)
        self.assertFalse(step.try_retire())
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            selected.close()
        self.assertIsNone(step._completion)
        self.assertEqual(removed, [])
        self.assertIs(selected.step, step)
        self.assertEqual(selected.worker.kv_recv_thread.read_lease_release, selected._release_native)
        with self.assertRaisesRegex(RuntimeError, "before native load"):
            selected.prepare(metadata, scheduled_tokens(metadata))

    def test_completion_query_error_poison_prevents_connector_reuse(self):
        selected, _, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        finish_readers(step)
        failure = RuntimeError("device query failed")

        class FailedQuery(Event):
            def query(self):
                raise failure

        completion = FailedQuery()
        with self.assertRaises(RuntimeError) as caught:
            step.finish(completion=completion)
        self.assertIs(caught.exception, failure)
        self.assertTrue(step._failed)
        self.assertFalse(step.finish(failed=True))  # Runtime failure escalation must succeed.
        for operation in (selected.begin_prepare, selected.close):
            with self.assertRaisesRegex(RuntimeError, "ownership retained"):
                operation()
        self.assertFalse(step.try_retire())
        self.assertIs(selected.step, step)
        self.assertEqual(completion.waits, 0)
        self.assertEqual(removed, [])

    def test_completion_wait_error_poison_prevents_next_prepare(self):
        selected, _, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        finish_readers(step)
        failure = RuntimeError("device synchronize failed")

        class FailedWait(Event):
            def synchronize(self):
                self.waits += 1
                raise failure

        completion = FailedWait(False)
        self.assertFalse(step.finish(completion=completion))
        with self.assertRaises(RuntimeError) as caught:
            selected.begin_prepare()
        self.assertIs(caught.exception, failure)
        self.assertTrue(step._failed)
        for operation in (selected.begin_prepare, selected.close):
            with self.assertRaisesRegex(RuntimeError, "ownership retained"):
                operation()
        self.assertEqual(completion.waits, 1)
        self.assertIs(selected.step, step)
        self.assertEqual(removed, [])

    def test_runtime_failure_escalation_after_release_error_is_terminal(self):
        selected, worker, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        finish_readers(step)
        releases = Mock(return_value=7)
        worker.m_store.batch_remove_lease = releases
        with self.assertRaisesRegex(RuntimeError, "read-lease release failed"):
            step.finish(completion=Event())
        # Backend refusal is not mislabeled as a device fault. An independent
        # caller may explicitly retry it, but a failed runtime must poison it.
        self.assertFalse(step._failed)
        self.assertFalse(step.finish(failed=True))
        releases.return_value = 0
        for operation in (selected.begin_prepare, selected.close):
            with self.assertRaisesRegex(RuntimeError, "ownership retained"):
                operation()
        self.assertFalse(step.try_retire())
        self.assertEqual(releases.call_count, 1)
        self.assertEqual(removed, [])
        with self.assertRaisesRegex(RuntimeError, "already finished"):
            step.finish(completion=Event())

    def test_source_does_not_install_or_replace_original_slot_waiter(self):
        selected, worker, _, _, exporter = fixture()
        selected.close()
        original = object()
        worker.external_slot_release_waiter = original
        worker.set_external_slot_release_waiter = Mock(side_effect=AssertionError("unexpected slot hook"))
        selected = source.SelectedKVSource(worker, exporter=exporter, event_factory=Event, upload=lambda x, d: x)
        selected.close()
        self.assertIs(worker.external_slot_release_waiter, original)
        worker.set_external_slot_release_waiter.assert_not_called()

    def test_missing_execution_completion_is_not_normal_close(self):
        selected, _, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        with self.assertRaisesRegex(TypeError, "real completion event"):
            step.finish()
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            selected.close()
        self.assertEqual(removed, [])

    def test_selected_shared_pages_stay_native_without_page_ownership(self):
        selected, worker, metadata, _, _ = fixture([request(), request("other", first=10)])
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertEqual([p.selected for p in step.requests], [True, True])
        # Request1 Q occupies513/514; it is not safe for selected request2
        # to claim those pages merely because their history occurs only once.
        omitted = np.r_[np.arange(1, 10), np.arange(515, 522)]
        pages = step.page_bases("layer.0.main")
        np.testing.assert_array_equal(np.flatnonzero(pages), omitted)
        shared = worker.layer_load_tasks[0][0].shared_block_data
        np.testing.assert_array_equal(shared.main_load_filter.keep_main, ~np.isin(shared.block_ids_arr, omitted))

    def test_all_shared_selected_pages_use_original_native_fastpath(self):
        selected, worker, metadata, _, exporter = fixture([request(), request("other")])
        before = worker.layer_load_tasks[0][0].shared_block_data
        self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
        self.assertEqual(exporter.calls, [])
        self.assertIs(worker.layer_load_tasks[0][0].shared_block_data, before)
        self.assertIsNone(before.main_load_filter)

    def test_high_id_native_kept_page_remains_in_zero_address_domain(self):
        req, full = request(), request("full", q=2048, first=3000)
        req.block_ids_by_group_np[0][511] = 3000
        req.block_ids_by_group[0] = req.block_ids_by_group_np[0].tolist()
        req.load_block_gvas_by_group_np[0][511] = full.load_block_gvas_by_group_np[0][0]
        selected, worker, metadata, _, _ = fixture([req, full])
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        for component in selected.components:
            pages = step.page_bases(component.layer_name)
            self.assertEqual(pages.shape, (3001,))
            self.assertEqual(pages[3000], 0)
            np.testing.assert_array_equal(np.flatnonzero(pages), np.arange(1, 512))
        shared = worker.layer_load_tasks[0][0].shared_block_data
        self.assertTrue(shared.main_load_filter.keep_main[shared.block_ids_arr == 3000].all())

    def test_actual_save_history_source_is_not_left_sparse(self):
        selected, worker, metadata, _, _ = fixture()
        original = worker.layer_save_tasks[0][0].shared_block_data
        saved = dataclasses.replace(
            original,
            block_ids_arr=np.r_[1, original.block_ids_arr],
            block_gvas_arr=np.r_[np.uint64(1 << 42), original.block_gvas_arr],
        )
        worker.layer_save_tasks = [[transfer_task(saved, i)] for i in range(worker.num_layers)]
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        for component in selected.components:
            self.assertEqual(step.page_bases(component.layer_name)[1], 0)
        load = worker.layer_load_tasks[0][0].shared_block_data
        self.assertTrue(load.main_load_filter.keep_main[load.block_ids_arr == 1].all())
        self.assertTrue(all(tasks[0].shared_block_data is saved for tasks in worker.layer_save_tasks))
        self.assertIsNone(saved.main_load_filter)

    def test_missing_native_layer_or_save_evidence_falls_back_before_publication(self):
        for fault in ("missing_layer", "missing_load_shared", "missing_save_shared", "missing_history_page"):
            with self.subTest(fault=fault):
                selected, worker, metadata, _, exporter = fixture()
                if fault == "missing_layer":
                    worker.layer_load_tasks[1] = []
                elif fault == "missing_load_shared":
                    worker.layer_load_tasks[1][0].shared_block_data = None
                elif fault == "missing_save_shared":
                    worker.layer_save_tasks[1][0].shared_block_data = None
                else:
                    shared = worker.layer_load_tasks[1][0].shared_block_data
                    worker.layer_load_tasks[1][0].shared_block_data = dataclasses.replace(
                        shared, block_ids_arr=shared.block_ids_arr[1:], block_gvas_arr=shared.block_gvas_arr[1:]
                    )
                    # Presence in save metadata must not be mistaken for load.
                    saved = worker.layer_save_tasks[0][0].shared_block_data
                    saved.block_ids_arr = np.r_[1, saved.block_ids_arr]
                    saved.block_gvas_arr = np.r_[np.uint64(1 << 42), saved.block_gvas_arr]
                with patch.object(selected, "_publish", wraps=selected._publish) as publish:
                    self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
                    publish.assert_not_called()
                self.assertEqual(exporter.calls, [])
                self.assertFalse(hasattr(metadata, "selected_kv_step"))
                for tasks in worker.layer_load_tasks:
                    for task in tasks:
                        if task.shared_block_data is not None:
                            self.assertIsNone(task.shared_block_data.main_load_filter)

    def test_same_group_distinct_shared_objects_each_need_coverage(self):
        selected, worker, metadata, _, _ = fixture()
        old = worker.layer_load_tasks[0][0].shared_block_data
        for i in range(worker.num_layers):
            worker.layer_load_tasks[i] = [
                transfer_task(
                    dataclasses.replace(
                        old, block_ids_arr=old.block_ids_arr.copy(), block_gvas_arr=old.block_gvas_arr.copy()
                    ),
                    i,
                )
            ]
        self.assertIsNotNone(selected.prepare(metadata, scheduled_tokens(metadata)))
        self.assertEqual(len({id(tasks[0].shared_block_data) for tasks in worker.layer_load_tasks}), worker.num_layers)

    def test_shared_coverage_scan_is_not_repeated_per_layer(self):
        selected, worker, metadata, _, _ = fixture()
        shared = worker.layer_load_tasks[0][0].shared_block_data
        calls = []
        original = source.np.isin

        def isin(elements, *args, **kwargs):
            if elements is shared.block_ids_arr:
                calls.append(True)
            return original(elements, *args, **kwargs)

        with patch.object(source.np, "isin", side_effect=isin):
            self.assertIsNotNone(selected.prepare(metadata, scheduled_tokens(metadata)))
        self.assertEqual(len(calls), 2)  # One coverage scan, one filter construction.

    def test_conflicting_native_kept_source_is_an_error_not_fallback(self):
        selected, worker, metadata, _, exporter = fixture([request(), request("full", q=2048, first=10)])
        shared = worker.layer_load_tasks[0][0].shared_block_data
        shared.block_gvas_arr[512] += np.uint64(1 << 30)  # Native request's first shared history page.
        with self.assertRaisesRegex(ValueError, "conflicting native load"):
            selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertIsNone(shared.main_load_filter)
        self.assertEqual(exporter.calls, [])

    def test_empty_saves_are_valid_and_next_step_waits_one_completion(self):
        selected, worker, metadata, removed, _ = fixture()
        worker.layer_save_tasks = [[] for _ in range(worker.num_layers)]
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        completion = Event(False)
        finish_readers(step)
        step.finish(completion=completion)
        selected.begin_prepare()
        self.assertEqual(completion.waits, 1)
        self.assertEqual(removed, [("selected",)])
        self.assertIsNone(selected.step)

    def test_scope_rejects_early_save_block_key_protocol(self):
        selected, worker, _, _, exporter = fixture()
        selected.close()
        worker.use_block_key_layerwise = True
        with self.assertRaisesRegex(ValueError, "GVA layerwise"):
            source.SelectedKVSource(worker, exporter=exporter)
        self.assertIsNone(worker.kv_recv_thread.read_lease_release)


class BatchAddressExporterTest(unittest.TestCase):
    def setUp(self):
        self.mapper_calls = []
        signature = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64))

        def mapper(address, memory_type, output):
            self.mapper_calls.append((address, memory_type))
            output[0] = address + 4096
            return 0

        self.exporter = address.HyBMAddressExporter(signature(mapper))
        self.gvas = np.array([8192], dtype=np.uint64)
        self.sizes = np.array([4096], dtype=np.uint64)

    def test_page_bytes_validation_precedes_offset_and_helper_checks(self):
        batch = self.exporter._batch = Mock(return_value=0)
        for page_bytes in (None, False, True, 0, -1, 1.0, np.int64(656)):
            for offset in (0, None):
                with (
                    self.subTest(page_bytes=page_bytes, offset=offset),
                    self.assertRaisesRegex(ValueError, "^page_bytes must be a positive integer$"),
                ):
                    self.exporter.export_component_arrays(self.gvas, self.sizes, offset, page_bytes)
        for page_bytes, offset in ((656, None), (656, True), (1 << 64, 0)):
            with (
                self.subTest(page_bytes=page_bytes, offset=offset),
                self.assertRaisesRegex(ValueError, "^Component offset and byte count must fit uint64$"),
            ):
                self.exporter.export_component_arrays(self.gvas, self.sizes, offset, page_bytes)
        batch.assert_not_called()
        self.assertEqual(self.mapper_calls, [])

    def test_missing_batch_helper_does_not_use_scalar_mapping(self):
        with self.assertRaisesRegex(RuntimeError, "batch address-export helper"):
            self.exporter.export_component_arrays(self.gvas, self.sizes, 0, 656)
        self.assertEqual(self.mapper_calls, [])

    def test_batch_errors_are_propagated_without_scalar_retry(self):
        for code, kind, message in (
            (1, ValueError, "leased allocation"),
            (2, RuntimeError, "No live device mapping"),
            (3, RuntimeError, "contiguous device byte span"),
        ):
            with self.subTest(code=code):
                batch = self.exporter._batch = Mock(return_value=code)
                with self.assertRaisesRegex(kind, message):
                    self.exporter.export_component_arrays(self.gvas, self.sizes, 0, 656)
                batch.assert_called_once()
                self.assertEqual(self.mapper_calls, [])

    def test_batch_library_load_failure_is_not_a_scalar_fallback(self):
        mapper = Mock()
        with (
            patch.object(
                address.ctypes,
                "CDLL",
                side_effect=[SimpleNamespace(hybm_gva_to_va=mapper), OSError("batch helper unavailable")],
            ) as load,
            self.assertRaisesRegex(OSError, "batch helper unavailable"),
        ):
            address.HyBMAddressExporter.from_loaded_library(__file__)
        self.assertEqual(load.call_count, 2)
        mapper.assert_not_called()


class SelectedStepLifetimeTest(unittest.TestCase):
    def make_step(self, keys=("a", "b", "a"), remove=None):
        removed = []
        origin = SimpleNamespace(
            worker=SimpleNamespace(
                m_store=SimpleNamespace(batch_remove_lease=remove or (lambda keys: removed.append(keys) or 0))
            ),
            components=(SimpleNamespace(layer_name="layer"),),
        )
        return source.SelectedLoadStep(
            origin, (), {"layer": object()}, Event(), keys, connector_metadata=SimpleNamespace()
        ), removed

    def test_key_iterable_is_snapshotted_once_in_first_seen_order(self):
        traversals = []

        class Keys:
            def __iter__(self):
                traversals.append(True)
                return iter(("b", "a", "b", "c", "a"))

        value, removed = self.make_step(keys=Keys())
        value.native_complete(["c", "b", "a"])
        self.assertTrue(value.finish(completion=Event()))
        self.assertEqual(traversals, [True])
        self.assertEqual(removed, [["b", "a", "c"]])

    def test_native_and_step_completion_both_hold_all_keys(self):
        value, removed = self.make_step()
        event = Event(False)
        value.native_complete(["a"])
        value.finish(completion=event)
        event.done = True
        self.assertFalse(value.try_retire())
        value.native_complete(["b"])
        self.assertTrue(value.try_retire())
        self.assertEqual(removed, [["a", "b"]])
        self.assertTrue(value.try_retire())

    def test_failed_backend_release_retains_completion_and_can_retry(self):
        result = [7]
        value, _ = self.make_step(keys=("a",), remove=lambda keys: result[0])
        event = Event()
        value.finish(completion=event)
        value.native_complete(["a"])
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            value.try_retire()
        self.assertIs(value._completion, event)
        self.assertFalse(value._closed)
        self.assertFalse(value._failed)
        value.wait_completion()
        self.assertEqual(event.waits, 1)
        result[0] = 0
        self.assertTrue(value.try_retire())
        value.wait_completion()
        self.assertEqual(event.waits, 1)

    def test_closed_step_does_not_synchronize_again(self):
        value, removed = self.make_step()
        value.native_complete(["a", "b"])
        completion = Event()
        self.assertTrue(value.finish(completion))
        with patch.object(completion, "synchronize", side_effect=AssertionError("retired source waited again")):
            value.wait_completion()
            value.wait_completion()
        self.assertEqual(removed, [["a", "b"]])

    def test_wait_failure_is_not_hidden_by_concurrent_retirement(self):
        entered, release = threading.Event(), threading.Event()
        failure = RuntimeError("device wait failed")
        errors = []

        class FailedWait(Event):
            def synchronize(self):
                entered.set()
                if not release.wait(1):
                    raise RuntimeError("Wait failure test did not continue")
                raise failure

        value, removed = self.make_step()
        self.assertFalse(value.finish(FailedWait()))  # Native readers still hold the lease.

        def wait():
            try:
                value.wait_completion()
            except BaseException as error:
                errors.append(error)

        waiter = threading.Thread(target=wait)
        waiter.start()
        try:
            self.assertTrue(entered.wait(1))
            value.native_complete(["a", "b"])
            self.assertTrue(value.try_retire())
        finally:
            release.set()
            waiter.join(1)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(errors, [failure])
        self.assertTrue(value._closed)
        self.assertTrue(value._failed)
        self.assertEqual(removed, [["a", "b"]])
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            value.wait_completion()

    def test_foreign_layer_keys_and_retired_callbacks_are_rejected(self):
        value, _ = self.make_step(keys=("a",))
        with self.assertRaisesRegex(ValueError, "layer does not belong"):
            value.page_bases("foreign")
        with self.assertRaisesRegex(ValueError, "actual layer"):
            value.native_layer_ready(1)
        with self.assertRaisesRegex(RuntimeError, "another lease"):
            value.native_complete(["foreign"])
        value.native_complete(["a"])
        value.finish(completion=Event())
        with self.assertRaisesRegex(RuntimeError, "retired"):
            value.native_complete(["a"])
        with self.assertRaisesRegex(RuntimeError, "retired"):
            value.page_bases("layer")

    def test_finish_is_once_and_success_requires_completion(self):
        value, removed = self.make_step()
        for invalid in (None, object(), SimpleNamespace(query=lambda: True)):
            with self.assertRaisesRegex(TypeError, "real completion"):
                value.finish(completion=invalid)
        value.finish(completion=Event())
        with self.assertRaisesRegex(RuntimeError, "already finished"):
            value.finish(completion=Event())
        self.assertEqual(removed, [])

    def test_failure_upgrade_is_one_way_and_cannot_reopen_closed_step(self):
        value, removed = self.make_step()
        value.finish(completion=Event(False))
        self.assertFalse(value.finish(failed=True))
        self.assertFalse(value.finish(failed=True))
        value.native_complete(["a", "b"])
        self.assertFalse(value.try_retire())
        with self.assertRaisesRegex(RuntimeError, "already finished"):
            value.finish(completion=Event())
        self.assertEqual(removed, [])
        closed, released = self.make_step()
        closed.native_complete(["a", "b"])
        self.assertTrue(closed.finish(completion=Event()))
        with self.assertRaisesRegex(RuntimeError, "already finished"):
            closed.finish(failed=True)
        self.assertEqual(released, [["a", "b"]])

    def test_failed_step_accepts_real_native_completion_but_never_retires(self):
        value, removed = self.make_step()
        value.finish(failed=True)
        value.native_complete(["a", "b"])
        self.assertEqual(value._native_done, {"a", "b"})
        self.assertFalse(value.try_retire())
        self.assertIsNone(value._completion)
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            value.wait_completion()
        self.assertEqual(removed, [])

    def test_device_wait_does_not_hold_native_callback_lock(self):
        value, removed = self.make_step()
        errors = []

        def complete_native():
            try:
                value.native_complete(["a", "b"])
            except BaseException as error:
                errors.append(error)

        callback = threading.Thread(target=complete_native)

        class WaitForCallback(Event):
            def synchronize(self):
                callback.start()
                callback.join(1)
                self.done = not callback.is_alive()

        event = WaitForCallback(False)
        value.finish(completion=event)
        value.wait_completion()
        callback.join(1)
        self.assertTrue(event.done, "Native callback was blocked by device wait")
        self.assertEqual(errors, [])
        self.assertTrue(value.try_retire())
        self.assertEqual(removed, [["a", "b"]])

    def test_concurrent_retirement_releases_keys_exactly_once(self):
        entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
        removed, results, errors = [], [], []

        def remove(keys):
            removed.append(keys)
            entered.set()
            if not release.wait(1):
                raise RuntimeError("Release test did not continue")
            return 0

        value, _ = self.make_step(remove=remove)
        value.finish(completion=Event())
        value.native_complete(["a", "b"])

        def retire():
            try:
                results.append(value.try_retire())
            except BaseException as error:
                errors.append(error)

        def second_retire():
            second_started.set()
            retire()

        first, second = threading.Thread(target=retire), threading.Thread(target=second_retire)
        first.start()
        self.assertTrue(entered.wait(1))
        second.start()
        self.assertTrue(second_started.wait(1))
        release.set()
        first.join(1)
        second.join(1)
        self.assertFalse(first.is_alive() or second.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(results, [True, True])
        self.assertEqual(removed, [["a", "b"]])


class NativeIntegrationTest(unittest.TestCase):
    def native_namespace(self):
        logger = SimpleNamespace(debug=lambda *args: None, warning=lambda *args: None, info=lambda *args: None)
        return {
            "np": np,
            "logger": logger,
            "LayerBatchReqMeta": type("LayerBatchReqMeta", (SimpleNamespace,), {}),
            "LayerRangeReqMeta": type("LayerRangeReqMeta", (SimpleNamespace,), {}),
        }

    def native_prepare(self, selected, worker):
        """Run the actual bind hook; only backend preparation is a CPU leaf."""
        namespace = {"reset_attention_compute_start_gate": Mock()}
        cls = load_methods("pool_worker.py", "KVPoolWorker", {"prepare_layerwise_step"}, namespace)
        infos = dict(worker._step_keyinfo_cache)
        worker.use_layerwise = True
        worker.selected_kv_source = selected
        worker._drain_deferred_last_save = Mock()

        def process(requests):
            self.assertIsNone(selected._prepared_metadata)
            self.assertEqual(selected._leased_infos, {})
            if requests:
                reset_native_tasks(worker, requests)
                selected.record_leased_infos(list(infos), list(infos.values()))
            else:
                worker.layer_load_tasks = [[] for _ in selected.components]
                worker.layer_save_tasks = [[] for _ in selected.components]

        worker.process_layer_data = Mock(side_effect=process)
        return lambda metadata: cls.prepare_layerwise_step(worker, metadata)

    def forward_binding(self):
        path = ROOT / "vllm_ascend/core/selected_kv_binding.py"
        tree = ast.parse(path.read_text())
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name in {"selected_kv_forward", "_finish_failed"}
        ]
        body = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
            type_ignores=[],
        )
        namespace = {"cast": cast, "contextmanager": contextmanager}
        exec(compile(ast.fix_missing_locations(body), str(path), "exec"), namespace)
        return namespace["selected_kv_forward"]

    def test_native_prepare_then_forward_binding_uses_live_q_before_first_load(self):
        requests = [request(), request("native", q=2048, first=1000)]
        selected, worker, metadata, removed, exporter = fixture(requests)
        prepare = self.native_prepare(selected, worker)
        prepare(metadata)
        self.assertEqual(exporter.calls, [])
        self.assertIs(selected._prepared_metadata, metadata)
        self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)
        runtime = SimpleNamespace(source=selected, source_step=None)
        batch = object()

        def bind(step, ids, histories, starts, attention):
            self.assertIs(step.connector_metadata, metadata)
            self.assertEqual(
                [(r.request_id, r.query_tokens) for r in step.requests], [("selected", 256), ("native", 2048)]
            )
            self.assertEqual(ids, ["native", "selected"])
            self.assertEqual(histories, [65536, 65536])
            self.assertEqual(starts, [0, 2048, 2304])
            runtime.source_step = step
            return batch

        def finish():
            finish_readers(runtime.source_step)
            runtime.source_step.finish(completion=Event())

        runtime.bind_step, runtime.finish_step = bind, finish
        runner = SimpleNamespace(
            selected_kv_runtime=runtime,
            input_batch=SimpleNamespace(
                num_reqs=2, req_ids=["native", "selected"], num_computed_tokens_cpu=[65536, 65536]
            ),
            query_start_loc=SimpleNamespace(cpu=[0, 2048, 2304]),
        )
        # No eager start_load is needed: save-only binding follows the same hook.
        output = SimpleNamespace(kv_connector_metadata=metadata, num_scheduled_tokens={"native": 2048, "selected": 256})
        with self.forward_binding()(runner, output, {}) as bound:
            self.assertIs(bound, batch)
            self.assertEqual(worker.next_layer_to_submit, 0)
            self.assertIsNone(selected._prepared_metadata)
            self.assertEqual(len(exporter.calls), 1)
            mask = worker.layer_load_tasks[0][0].shared_block_data.main_load_filter.keep_main
            self.assertFalse(mask[:512].any())
            self.assertTrue(mask[512:].all())
            worker.next_layer_to_submit = 1  # First model hook may submit only now.
        self.assertEqual(removed, [("selected", "native")])

        # An empty/no-forward bind still retires the source and clears its filter.
        empty = SimpleNamespace(requests=[])
        prepare(empty)
        self.assertIsNone(selected.step)
        self.assertIs(selected._prepared_metadata, empty)
        # The next step may reuse request IDs but must bind its own metadata.
        next_metadata = SimpleNamespace(requests=requests)
        prepare(next_metadata)
        self.assertIs(selected._prepared_metadata, next_metadata)
        self.assertIsNone(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter)
        self.assertEqual(worker.next_layer_to_submit, 0)
        with self.assertRaisesRegex(RuntimeError, "native preparation of this connector metadata"):
            selected.prepare(metadata, output.num_scheduled_tokens)
        next_step = selected.prepare(next_metadata, output.num_scheduled_tokens)
        self.assertIsNotNone(next_step)
        finish_readers(next_step)
        next_step.finish(completion=Event())
        selected.close()

    def test_failed_native_prepare_cannot_publish_or_reuse_a_previous_identity(self):
        selected, worker, metadata, _, exporter = fixture()
        prepare = self.native_prepare(selected, worker)
        next_metadata = SimpleNamespace(requests=metadata.requests)
        worker.process_layer_data.side_effect = RuntimeError("native task preparation failed")
        with self.assertRaisesRegex(RuntimeError, "native task preparation failed"):
            prepare(next_metadata)
        self.assertIsNone(selected._prepared_metadata)
        self.assertEqual(selected._leased_infos, {})
        for value in (metadata, next_metadata):
            with self.assertRaisesRegex(RuntimeError, "native preparation of this connector metadata"):
                selected.prepare(value, scheduled_tokens(value))
        self.assertEqual(exporter.calls, [])
        # A real new successful native preparation can retry; no old proof survives.
        self.native_prepare(selected, worker)(next_metadata)
        step = selected.prepare(next_metadata, scheduled_tokens(next_metadata))
        self.assertIsNotNone(step)
        finish_readers(step)
        step.finish(completion=Event())
        selected.close()

    def test_forward_binding_cannot_run_without_successful_native_prepare(self):
        selected, worker, metadata, _, exporter = fixture()
        selected.begin_prepare()
        runtime = SimpleNamespace(source=selected, source_step=None, bind_step=Mock(), finish_step=Mock())
        runner = SimpleNamespace(selected_kv_runtime=runtime)
        output = SimpleNamespace(kv_connector_metadata=metadata, num_scheduled_tokens=scheduled_tokens(metadata))
        with (
            self.assertRaisesRegex(RuntimeError, "native preparation of this connector metadata"),
            self.forward_binding()(runner, output, {}),
        ):
            self.fail("The model must not run with unprepared native tasks")
        runtime.bind_step.assert_not_called()
        runtime.finish_step.assert_not_called()
        self.assertEqual(exporter.calls, [])

    def test_full_shared_prefix_keeps_native_tasks_after_a_filtered_step(self):
        selected, worker, metadata, removed, _ = fixture()
        first = selected.prepare(metadata, scheduled_tokens(metadata))
        old_tasks = worker.layer_load_tasks
        self.assertIsNotNone(old_tasks[0][0].shared_block_data.main_load_filter)
        finish_readers(first)
        first.finish(completion=Event())
        selected.begin_prepare()

        metadata.requests.append(request("full", q=2048))
        infos = dict(worker._step_keyinfo_cache)
        infos["full"] = infos["selected"]
        namespace = self.native_namespace()
        namespace.update(SharedBlockData=Shared, KVCacheStoreLayerRecvingThread=SimpleNamespace)
        builder_cls = load_methods(
            "kv_transfer.py",
            "LayerBatchBuilder",
            {
                "_ensure_buf",
                "_dedupe_transfer_blocks",
                "_require_request_arrays",
                "build_shared",
                "_build_transfer_arrays",
                "build_addrs",
            },
            namespace,
        )
        builder = builder_cls()
        builder.__dict__.update(vars(worker.kv_recv_thread.layer_batch_builder))
        builder.group_id = 0
        builder.page_size_bytes = int(builder._block_len_np.sum())
        builder._block_ids_buf = builder._block_gvas_buf = None
        worker.kv_recv_thread.build_shared_data = lambda task: builder.build_shared(task, is_save=False)
        pool_cls = load_methods(
            "pool_worker.py", "KVPoolWorker", {"process_layer_data", "_build_shared_load_data"}, namespace
        )
        worker._masked_runs_cache = {}
        worker._compute_reachable_store_masks = lambda req: None
        worker._groups_for_layerwise_transfer = lambda layer_id: [(0, layer_id)]
        worker._process_save_for_layer_batch = Mock()
        worker._prepare_load_gvas = lambda requests: worker._step_lease_result.update({r.req_id: 0 for r in requests})
        worker._alloc_gvas_for_save = worker._build_shared_save_data = Mock()
        worker._build_shared_load_data = lambda: pool_cls._build_shared_load_data(worker)
        for req in metadata.requests:
            req.save_keys = []

        def populate(requests, layer_id, group_id, _layer_idx):
            ranges = [
                SimpleNamespace(request=req, start_block=0, end_block=512, partial_block_index=None) for req in requests
            ]
            worker.layer_load_tasks[layer_id].append(
                SimpleNamespace(
                    group_id=group_id,
                    layer_idx_in_group=layer_id,
                    block_ranges=ranges,
                    use_key_major_ranges=False,
                    shared_block_data=None,
                )
            )

        worker._process_load_for_layer_batch = populate
        pool_cls.process_layer_data(worker, metadata.requests)
        selected.record_leased_infos(list(infos), list(infos.values()))
        selected.native_prepared(metadata)
        tasks = tuple(task for layer in worker.layer_load_tasks for task in layer)
        shared = tasks[0].shared_block_data
        self.assertIsNot(worker.layer_load_tasks, old_tasks)
        self.assertIsNot(shared, old_tasks[0][0].shared_block_data)
        before = [builder.build_addrs(shared, layer_id) for layer_id in range(worker.num_layers)]

        step = selected.prepare(metadata, scheduled_tokens(metadata))

        self.assertIsNone(step)
        self.assertFalse(hasattr(metadata, "selected_kv_step"))
        self.assertIsNone(shared.main_load_filter)
        for layer_id, task in enumerate(tasks):
            self.assertIs(worker.layer_load_tasks[layer_id][0], task)
            self.assertIs(task.shared_block_data, shared)
            after = builder.build_addrs(task.shared_block_data, layer_id)
            for name in ("addr_array", "size_array", "gvas_array"):
                np.testing.assert_array_equal(getattr(after, name), getattr(before[layer_id], name))
            self.assertEqual(after.load_keys, ["selected", "full"])
        self.assertEqual(removed, [("selected",)])
        worker.kv_recv_thread.read_lease_release(["selected", "full"])
        self.assertEqual(removed, [("selected",), ("selected", "full")])

    def test_real_native_builder_preserves_all_surviving_addresses_and_save_rows(self):
        selected, worker, metadata, _, _ = fixture()
        selected.prepare(metadata, scheduled_tokens(metadata))
        namespace = self.native_namespace()
        cls = load_methods("kv_transfer.py", "LayerBatchBuilder", {"_build_transfer_arrays", "build_addrs"}, namespace)
        builder = cls()
        builder.__dict__.update(vars(worker.kv_recv_thread.layer_batch_builder))
        builder.page_size_bytes = int(builder._block_len_np.sum())
        shared = worker.layer_load_tasks[0][0].shared_block_data
        original = dataclasses.replace(shared, main_load_filter=None)
        before, after = builder.build_addrs(original, 1), builder.build_addrs(shared, 1)
        count = original.block_ids_arr.size
        keep = np.ones((count, 3), dtype=bool)
        keep[:512, 0] = False
        for name in ("addr_array", "size_array", "gvas_array"):
            np.testing.assert_array_equal(getattr(after, name), getattr(before, name).reshape(count, 3)[keep])
        saved = builder.build_addrs(original, 1)
        np.testing.assert_array_equal(saved.addr_array, before.addr_array)
        np.testing.assert_array_equal(saved.gvas_array, before.gvas_array)
        self.assertEqual(after.load_keys, before.load_keys)

    def test_mixed_native_only_shared_object_bypasses_filter_and_extra_array_copies(self):
        requests = [request(), request("full", q=2048, first=2049)]
        selected, worker, metadata, _, _ = fixture(requests)
        combined = worker.layer_load_tasks[0][0].shared_block_data
        selected_shared = dataclasses.replace(
            combined, block_ids_arr=combined.block_ids_arr[:512], block_gvas_arr=combined.block_gvas_arr[:512]
        )
        native_shared = dataclasses.replace(
            combined, block_ids_arr=combined.block_ids_arr[512:], block_gvas_arr=combined.block_gvas_arr[512:]
        )
        for layer in range(worker.num_layers):
            worker.layer_load_tasks[layer] = [
                transfer_task(selected_shared, layer),
                transfer_task(native_shared, layer),
            ]
        namespace = self.native_namespace()
        cls = load_methods("kv_transfer.py", "LayerBatchBuilder", {"_build_transfer_arrays", "build_addrs"}, namespace)
        builder = cls()
        builder.__dict__.update(vars(worker.kv_recv_thread.layer_batch_builder))
        builder.page_size_bytes = int(builder._block_len_np.sum())
        original_arrays = []
        real_build = builder._build_transfer_arrays

        def build(*args):
            arrays = real_build(*args)
            original_arrays.append(arrays)
            return arrays

        step = selected.prepare(metadata, scheduled_tokens(metadata))

        self.assertEqual([r.selected for r in step.requests], [True, False])
        with patch.object(builder, "_build_transfer_arrays", side_effect=build):
            for layer, tasks in enumerate(worker.layer_load_tasks):
                self.assertIsNot(tasks[0].shared_block_data, selected_shared)
                self.assertIsNotNone(tasks[0].shared_block_data.main_load_filter)
                self.assertIs(tasks[1].shared_block_data, native_shared)
                self.assertIsNone(native_shared.main_load_filter)
                result = builder.build_addrs(native_shared, layer)
                for name, array in zip(("addr_array", "size_array", "gvas_array"), original_arrays[-1]):
                    self.assertIs(getattr(result, name), array)

    def test_native_scheduler_adds_no_selection_fields_for_any_request_set(self):
        class Metadata:
            def __init__(self, *args):
                self.requests = []

            def add_request(self, value):
                self.requests.append(value)

        namespace = {"AscendConnectorMetadata": Metadata}
        cls = load_methods("pool_scheduler.py", "KVPoolScheduler", {"build_connector_meta"}, namespace)
        for represented, scheduled in (
            (("a", "b"), {"a": 128, "b": 2048}),
            (("a",), {"a": 128, "b": 2048}),
            (("a", "extra"), {"a": 128}),
        ):
            with self.subTest(represented=represented, scheduled=scheduled):
                scheduler = cls()
                scheduler.kv_role, scheduler.consumer_is_to_put, scheduler.can_put = "kv_producer", False, True
                scheduler._request_trackers, scheduler._unfinished_requests = {}, {}
                scheduler._preempted_req_ids, scheduler._loading_req_ids = set(), set()
                scheduler._process_new_request = lambda value, *args: value
                scheduler.touch_sending_mamba_blocks = lambda value: None
                output = SimpleNamespace(
                    finished_req_ids=set(),
                    preempted_req_ids=set(),
                    num_scheduled_tokens=scheduled,
                    scheduled_new_reqs=[SimpleNamespace(req_id=name) for name in represented],
                    scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
                )
                metadata = scheduler.build_connector_meta(output)
                self.assertFalse(hasattr(metadata, "scheduled_requests_complete"))
                for item in metadata.requests:
                    self.assertFalse(hasattr(item, "num_scheduled_tokens"))

    def test_native_scheduler_keeps_only_original_q_lookup(self):
        class Metadata:
            def __init__(self, *args):
                self.requests = []

            def add_request(self, value):
                self.requests.append(value)

        class ScheduledTokens(dict):
            def __init__(self):
                super().__init__(a=2)
                self.lookups = []

            def __getitem__(self, key):
                self.lookups.append(key)
                return super().__getitem__(key)

            def get(self, *args):
                raise AssertionError("Disabled selection must not perform its extra per-request Q lookup")

            def items(self):
                raise AssertionError("Disabled selection must not build its request-set guard")

        namespace = {
            "AscendConnectorMetadata": Metadata,
            "RequestTracker": SimpleNamespace,
            "normalize_block_ids_by_group": lambda value: [value],
            "_new_req_prefill_tokens": lambda value: value.prompt_token_ids,
        }
        cls = load_methods(
            "pool_scheduler.py", "KVPoolScheduler", {"build_connector_meta", "_process_new_request"}, namespace
        )
        scheduler = cls()
        scheduler.kv_role, scheduler.consumer_is_to_put, scheduler.can_put = "kv_producer", False, True
        scheduler._request_trackers, scheduler.load_specs = {}, {}
        scheduler._preempted_req_ids, scheduler._loading_req_ids = set(), set()
        scheduler.enable_kv_events = False
        scheduler.num_speculative_blocks_by_group, scheduler.grouped_block_size = [0], [128]
        scheduler.touch_sending_mamba_blocks = lambda value: None
        scheduler._build_req_meta = lambda tracker, *args: SimpleNamespace(req_id=tracker.req_id)
        request = SimpleNamespace(req_id="a", num_computed_tokens=0, block_ids=[1], prompt_token_ids=[1, 2])
        scheduler._unfinished_requests = {"a": (SimpleNamespace(block_hashes=[]), [1])}
        scheduled = ScheduledTokens()
        output = SimpleNamespace(
            finished_req_ids=set(),
            preempted_req_ids=set(),
            num_scheduled_tokens=scheduled,
            scheduled_new_reqs=[request],
            scheduled_cached_reqs=SimpleNamespace(req_ids=[]),
        )
        metadata = scheduler.build_connector_meta(output)
        self.assertEqual(scheduled.lookups, ["a"])
        self.assertFalse(hasattr(metadata.requests[0], "num_scheduled_tokens"))
        self.assertFalse(hasattr(metadata, "scheduled_requests_complete"))
        self.assertFalse(hasattr(metadata, "selected_kv_step"))

    def test_native_empty_save_records_event_before_cpu_completion(self):
        namespace = self.native_namespace()
        namespace.update(cast=lambda cls, value: value, KVCacheStoreLayerSendingThreadBase=object)
        cls = load_methods("pool_worker.py", "KVPoolWorker", {"save_kv_layer"}, namespace)
        event = Event(False)

        class Finished:
            def set(self):
                self.recorded = event.recorded

        finished = Finished()
        worker = SimpleNamespace(
            current_layer=0,
            num_layers=3,
            use_block_key_layerwise=False,
            block_key_hybrid=False,
            sync_save_events=[event],
            layer_save_finished_events=[finished],
            kv_send_thread=SimpleNamespace(raise_if_failed=lambda: None),
            _attention_saved_layers=set(),
            layer_save_tasks=[[]],
        )
        cls.save_kv_layer(worker, object())
        self.assertTrue(finished.recorded)
        self.assertFalse(event.done)
        self.assertEqual(worker.current_layer, 1)

    def test_native_empty_receive_still_synchronizes_same_slot_save(self):
        namespace = self.native_namespace()
        cls = load_methods("kv_transfer.py", "KVCacheStoreLayerRecvingThread", {"_handle_request"}, namespace)
        recv = cls()
        event = Event(False)
        saved = threading.Event()
        saved.set()
        recv.layer_save_finished_events = [saved]
        recv.sync_save_events = [event]
        recv.save_failure_checker = None
        recv.external_slot_release_waiter = None
        recv.selected_indexer_load = None
        recv.layer_load_finished_events = [threading.Event() for _ in range(3)]
        finished = []
        recv.request_queue = SimpleNamespace(task_done=lambda: finished.append(True))
        recv._handle_request(
            SimpleNamespace(wait_for_save_layer=0, transfer_tasks=[], layer_id=2, attention_start_gate=None)
        )
        self.assertEqual(event.waits, 1)
        self.assertTrue(event.done)
        self.assertFalse(saved.is_set())
        self.assertTrue(recv.layer_load_finished_events[2].is_set())
        self.assertEqual(finished, [True])

    def test_real_receiver_zero_copy_still_reports_native_lease_completion(self):
        selected, _, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        namespace = self.native_namespace()
        cls = load_methods("kv_transfer.py", "KVCacheStoreLayerRecvingThread", {"_handle_request"}, namespace)
        base = load_methods("kv_transfer.py", "KVTransferThread", {"_batch_copy_with_limits"}, namespace)
        recv = cls()
        copies, acknowledged, finished = [], [], []
        recv.m_store = SimpleNamespace(store=SimpleNamespace(batch_copy=lambda *args: copies.append(args) or 0))
        recv._batch_copy_with_limits = base._batch_copy_with_limits.__get__(recv)
        recv.final_layer_id = 2
        recv.external_slot_release_waiter = None
        recv.max_transfer_blocks = recv.max_transfer_bytes = 0
        recv._stagger_h2d_submit = lambda layer: None
        recv.layer_load_finished_events = [threading.Event() for _ in range(3)]
        recv.request_queue = SimpleNamespace(task_done=lambda: acknowledged.append(True))
        recv.get_event = threading.Event()
        recv.set_finished_request = finished.append
        recv.read_lease_release = selected._release_native
        empty = np.empty(0, dtype=np.int64)
        built = namespace["LayerBatchReqMeta"](
            req_ids=["selected"],
            is_last_chunks=[True],
            load_keys=["selected"],
            addr_array=empty,
            size_array=empty,
            gvas_array=empty,
        )
        recv.group_builders = [SimpleNamespace(build_addrs=lambda shared, layer: built)]
        task = SimpleNamespace(shared_block_data=object(), group_id=0, layer_idx_in_group=2, use_key_major_ranges=False)
        data = SimpleNamespace(wait_for_save_layer=None, transfer_tasks=[task], layer_id=2, attention_start_gate=None)
        mte_loads = []

        def copy_indexer(layer):
            self.assertFalse(recv.layer_load_finished_events[layer].is_set())
            self.assertEqual(removed, [])
            mte_loads.append(layer)

        recv.selected_indexer_load = copy_indexer
        recv._handle_request(data)
        self.assertEqual(mte_loads, [2])
        self.assertEqual(copies, [])
        self.assertEqual(acknowledged, [True])
        self.assertEqual(finished, ["selected"])
        self.assertEqual(removed, [])
        finish_readers(step)
        step.finish(completion=Event())
        self.assertEqual(removed, [("selected",)])

        recv.layer_load_finished_events[2].clear()
        recv.selected_indexer_load = Mock(side_effect=RuntimeError("MTE completion failed"))
        with self.assertRaisesRegex(RuntimeError, "MTE completion failed"):
            recv._handle_request(data)
        self.assertFalse(recv.layer_load_finished_events[2].is_set())
        self.assertEqual(acknowledged, [True])

    def test_real_worker_empty_final_layer_retires_last_native_reader(self):
        selected, worker, metadata, removed, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        step.native_layer_ready(0)
        step.native_layer_ready(1)
        namespace = self.native_namespace()
        namespace.update(
            reset_attention_compute_start_gate=lambda: object(), is_kv_save_role=lambda role, consumer: True
        )
        cls = load_methods("pool_worker.py", "KVPoolWorker", {"wait_for_layer_load"}, namespace)
        worker.current_layer = 2
        worker.layer_load_finished_events = [threading.Event() for _ in range(3)]
        worker.kv_recv_thread.raise_if_failed = lambda: None
        worker.layer_load_tasks[2] = []
        worker.prefetch_layer_map = {}
        worker._submit_ready_layer_loads = lambda: None
        worker.selected_kv_source = selected
        cls.wait_for_layer_load(worker)
        step.finish(completion=Event())
        self.assertEqual(removed, [("selected",)])


class NativeAddressDtypeTest(unittest.TestCase):
    def test_signed_native_metadata_reaches_the_real_uint64_export_guard(self):
        signature = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint64, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64))
        mapper = signature(lambda *_: 0)
        exporter = address.HyBMAddressExporter(mapper)

        def batch(_mapper, gvas, _sizes, count, _offset, _bytes, output, _failed, _status):
            for index in range(count):
                output[index] = gvas[index] + (1 << 40)
            return 0

        exporter._batch = batch
        for history, queries in ((20480, (128,)), (65536, (128, 983)), (262144, (128, 1024))):
            for query in queries:
                with self.subTest(H=history, Q=query):
                    req = request(h=history, q=query)
                    req.load_block_gvas_by_group_np = [req.load_block_gvas_by_group_np[0].astype(np.int64)]
                    selected, _, metadata, _, _ = fixture([req])
                    selected.exporter = exporter
                    step = selected.prepare(metadata, scheduled_tokens(metadata))
                    self.assertIsNotNone(step)
                    ids = req.block_ids_by_group_np[0][: history // 128]
                    for component in selected.components:
                        expected = req.load_block_gvas_by_group_np[0] + (1 << 40) + component.object_offset
                        np.testing.assert_array_equal(step.page_bases(component.layer_name)[ids], expected)


class IndexerMTECopyTest(unittest.TestCase):
    def test_original_views_and_offsets_are_used(self):
        selected, worker, metadata, _, exporter = fixture(indexers=True)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertTrue(step.indexer_mte)
        builder = worker.kv_recv_thread.layer_batch_builder
        for layer, component in enumerate(selected.indexers):
            self.assertEqual(component.components, (1, 2))
            self.assertIs(component.caches[0], worker.kv_caches[f"layer.{layer}.indexer"][0])
            for offset, column in zip(component.offsets, (1, 2)):
                self.assertEqual(offset, 512 + int(builder._block_len_np[: layer * 3 + column].sum()))
        self.assertEqual(exporter.calls[0][-1], 512 + int(builder._block_len_np.sum()))
        self.assertIs(worker.kv_recv_thread.selected_indexer_load.__self__, selected)

    def test_partial_shared_native_pages_and_saves_stay_sdk(self):
        demand = request(h=65537, q=127)
        native = request("native", h=65536, q=2048, first=1024)
        selected, worker, metadata, _, _ = fixture([demand, native], indexers=True)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        shared = worker.layer_load_tasks[0][0].shared_block_data
        values = np.arange(shared.block_ids_arr.size * 3).reshape(-1, 3)
        actual = shared.main_load_filter.apply(0, values.ravel(), values.ravel(), values.ravel())[0]
        np.testing.assert_array_equal(actual, values[shared.main_load_filter.keep_main].ravel())
        partial_id = demand.block_ids_by_group_np[0][512]
        self.assertEqual(step.page_bases("layer.0.main")[partial_id], 0)
        self.assertTrue(
            shared.main_load_filter.keep_main[shared.block_ids_arr >= native.block_ids_by_group_np[0][0]].all()
        )
        self.assertIsNone(worker.layer_save_tasks[0][0].shared_block_data.main_load_filter)

    def test_backend_choice_is_frozen_for_this_step(self):
        selected, worker, metadata, _, _ = fixture(indexers=True)
        selected.indexer_mte_enabled = False
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        selected.indexer_mte_enabled = True
        self.assertFalse(step.indexer_mte)
        shared = worker.layer_load_tasks[0][0].shared_block_data
        self.assertEqual(shared.main_load_filter.indexer_components, ())
        # No NPU exists in this CPU test: the frozen SDK choice must return.
        selected.load_indexer(0)

    def test_device_handoff_blocks_consumers_and_retirement_until_copy(self):
        for mixed in (False, True):
            with self.subTest(mixed=mixed):
                self._check_device_handoff(mixed)

    def _check_device_handoff(self, mixed):
        requests = [request()]
        if mixed:
            requests.append(request("native", q=2048, first=1024))
        selected, worker, metadata, removed, _ = fixture(requests, indexers=True)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        copied, consumed = [], []

        class Queue:
            def __init__(self):
                self.tasks = []

            def wait_event(self, event):
                self.tasks.append(lambda: event.done)

            def submit(self, action):
                self.tasks.append(lambda: (action(), True)[1])

            def run(self):
                while self.tasks and self.tasks[0]():
                    self.tasks.pop(0)

        transfer, compute = Queue(), Queue()

        class Done(Event):
            def __init__(self):
                super().__init__(False)

            def record(self):
                transfer.submit(lambda: setattr(self, "done", True))

            def query(self):
                raise AssertionError("LI events must not be polled at retirement")

            def synchronize(self):
                raise AssertionError("LI submission must not block the CPU")

        @contextmanager
        def stream(value):
            self.assertIs(value, selected._metadata_stream)
            yield

        def copy(*args):
            layer = len(step._indexer_events)
            self.assertIs(args[0], step.page_bases(selected.components[layer].layer_name))
            self.assertEqual(args[3:], (selected.components[layer].object_offset, *selected.indexers[layer].offsets))
            transfer.submit(lambda: copied.append(layer))

        selected._event = Done
        fake = SimpleNamespace(
            npu=SimpleNamespace(stream=stream, current_stream=lambda: compute),
            ops=SimpleNamespace(dsa_prefill_queue=SimpleNamespace(li_page_gather=copy)),
        )
        with patch.object(source, "torch", fake):
            for layer in range(worker.num_layers):
                selected.load_indexer(layer)
                selected.native_layer_ready(layer)
                selected.native_layer_ready(layer)  # Native notification stays idempotent.
                compute.submit(lambda layer=layer: consumed.append((layer, layer in copied)))
        done = Event(False)
        compute.submit(lambda: setattr(done, "done", True))
        step.native_complete([r.req_id for r in requests])
        self.assertFalse(step.finish(done))
        compute.run()
        self.assertEqual(consumed, [])
        self.assertEqual(removed, [])
        self.assertFalse(step.try_retire())
        transfer.run()
        compute.run()
        self.assertEqual(consumed, [(0, True), (1, True), (2, True)])
        self.assertTrue(step.try_retire())
        self.assertEqual(removed, [tuple(r.req_id for r in requests)])
        self.assertEqual(len(step._indexer_events), worker.num_layers)
        selected.begin_prepare()
        reset_native_tasks(worker, requests)
        infos = worker._step_keyinfo_cache
        selected.record_leased_infos(list(infos), list(infos.values()))
        selected.native_prepared(metadata)
        next_step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertIsNot(next_step, step)
        self.assertEqual(next_step._indexer_events, {})
        self.assertEqual(next_step._native_layers, set())

    def test_unconsumed_copy_cannot_retire_even_after_native_callback(self):
        selected, _, metadata, removed, _ = fixture(indexers=True)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        step.record_indexer_load(0, Event())
        step.native_complete(["selected"])
        with self.assertRaisesRegex(RuntimeError, "not joined"):
            step.finish(Event())
        self.assertFalse(step.try_retire())
        self.assertTrue(step._failed)
        self.assertEqual(removed, [])
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            selected.begin_prepare()

    def test_wait_failure_does_not_mark_owner_ready(self):
        selected, _, metadata, removed, _ = fixture(indexers=True)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        with self.assertRaisesRegex(RuntimeError, "event is missing"):
            step.native_layer_ready(0)
        step.record_indexer_load(0, Event(False))
        stream = SimpleNamespace(wait_event=Mock(side_effect=RuntimeError("wait enqueue failed")))
        with (
            patch.object(source, "torch", SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: stream))),
            self.assertRaisesRegex(RuntimeError, "wait enqueue failed"),
        ):
            step.native_layer_ready(0)
        self.assertNotIn(0, step._native_layers)
        step.finish(failed=True)
        self.assertEqual(removed, [])
        self.assertIn(0, step._indexer_events)

    def test_copy_or_event_record_failure_retains_source(self):
        for failure in ("launch", "record"):
            with self.subTest(failure=failure):
                selected, _, metadata, removed, _ = fixture(indexers=True)
                step = selected.prepare(metadata, scheduled_tokens(metadata))

                class FailedRecord(Event):
                    def record(self):
                        raise RuntimeError("record failed")

                @contextmanager
                def stream(value):
                    yield

                selected._event = FailedRecord
                launch = Mock(side_effect=RuntimeError("launch failed") if failure == "launch" else None)
                fake = SimpleNamespace(
                    npu=SimpleNamespace(stream=stream),
                    ops=SimpleNamespace(dsa_prefill_queue=SimpleNamespace(li_page_gather=launch)),
                )
                with patch.object(source, "torch", fake), self.assertRaisesRegex(RuntimeError, failure + " failed"):
                    selected.load_indexer(0)
                self.assertEqual(set(step._indexer_events), {0} if failure == "record" else set())
                self.assertTrue(step._failed)
                step.finish(failed=True)
                self.assertFalse(step.try_retire())
                self.assertEqual(removed, [])
                self.assertIs(selected.step, step)

    def test_late_device_failure_retains_all_joined_events(self):
        selected, worker, metadata, removed, _ = fixture(indexers=True)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        stream = SimpleNamespace(wait_event=Mock())
        with patch.object(source, "torch", SimpleNamespace(npu=SimpleNamespace(current_stream=lambda: stream))):
            for layer in range(worker.num_layers):
                step.record_indexer_load(layer, Event(False))
                step.native_layer_ready(layer)
        final = SimpleNamespace(query=Mock(side_effect=RuntimeError("device failure")), synchronize=Mock())
        with self.assertRaisesRegex(RuntimeError, "device failure"):
            step.finish(final)
        self.assertTrue(step._failed)
        self.assertEqual(len(step._indexer_events), worker.num_layers)
        self.assertEqual(removed, [])
        self.assertFalse(step.finish(failed=True))

    def test_foreign_and_duplicate_publication_is_rejected(self):
        selected, _, metadata, _, _ = fixture(indexers=True)
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        first = Event(False)
        step.record_indexer_load(0, first)
        for invalid in (0, -1, 3, True):
            with self.subTest(layer=invalid), self.assertRaisesRegex(ValueError, "another layer"):
                step.record_indexer_load(invalid, Event())
        self.assertIs(step._indexer_events[0], first)

    def test_native_only_keeps_original_no_event_path(self):
        selected, _, metadata, removed, _ = fixture([request(q=2048)], indexers=True)
        self.assertIsNone(selected.prepare(metadata, scheduled_tokens(metadata)))
        selected.load_indexer(0)
        selected.native_layer_ready(0)
        self.assertEqual(removed, [])

    def test_missing_or_nonstandard_indexer_never_suppresses_sdk(self):
        selected, worker, metadata, _, _ = fixture()
        step = selected.prepare(metadata, scheduled_tokens(metadata))
        self.assertFalse(step.indexer_mte)
        self.assertEqual(worker.layer_load_tasks[0][0].shared_block_data.main_load_filter.indexer_components, ())
        selected, worker, _, _, _ = fixture(indexers=True)
        worker.kv_recv_thread.layer_batch_builder._block_stride_np[1] += 1
        with self.assertRaisesRegex(ValueError, "LI page"):
            selected._indexer_components()


if __name__ == "__main__":
    unittest.main()
