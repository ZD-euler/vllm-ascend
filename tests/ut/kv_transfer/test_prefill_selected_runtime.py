# SPDX-License-Identifier: Apache-2.0
"""CPU integration/lifetime contracts; NPU copy, union and SFA remain unverified."""

import ast
import builtins
import ctypes
import importlib.util
import sys
import unittest
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
DIRECTORY = ROOT / "vllm_ascend/distributed/kv_transfer/sparse_kv_offload"
PACKAGE = "vllm_ascend.distributed.kv_transfer.sparse_kv_offload"


def load_module(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", DIRECTORY / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


load_module("prefill_selected_plan")
workspace = load_module("prefill_selected_workspace")
runtime = load_module("prefill_selected_runtime")


class RuntimeImportTest(unittest.TestCase):
    def test_boundary_types_do_not_import_runner_attention_or_connector(self):
        original_import = builtins.__import__
        type_only_modules = {
            "vllm_ascend.attention.sfa_v1",
            "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.selected_kv_source",
            "vllm_ascend.worker.model_runner_v1",
        }

        def guarded_import(name, *args, **kwargs):
            if name in type_only_modules:
                self.fail(f"Runtime boundary types imported {name}")
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules), patch.object(builtins, "__import__", guarded_import):
            load_module("prefill_selected_runtime")


class Event:
    def __init__(self):
        self.done, self.recorded, self.waits = True, False, 0

    def record(self):
        self.recorded = True

    def query(self):
        return self.done

    def synchronize(self):
        self.waits += 1
        self.done = True


class Planner:
    """Mock only the native kernel boundary; retain actual tensor identities."""

    def __init__(self, device, capacity, query_capacity, max_requests):
        self.capacity, self.query_capacity, self.max_requests = capacity, query_capacity, max_requests
        self.outputs = (object(), object())
        self.build = Mock(return_value=self.outputs)
        self.load = Mock()
        self.release_quiescent, self.close_quiescent = Mock(), Mock()


class Step:
    def __init__(self, requests):
        self.requests = requests
        self.finish = Mock(side_effect=self._finish)
        self.completion = None
        self.try_retire = Mock(return_value=True)
        self.main_copy_submitted = Mock()
        self.source_ready = Event()
        self.source = torch.zeros(8193, dtype=torch.uint64)

    def ready(self, name):
        return self.source_ready

    def page_bases(self, name):
        return self.source

    def _finish(self, completion=None, failed=False):
        self.completion = completion

    def wait_completion(self):
        if self.completion is None:
            raise RuntimeError("Selected source has not finished")
        self.completion.synchronize()


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        empty_like = torch.empty_like

        def cpu_empty_like(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty_like(*args, **kwargs)

        self.stream = NS(npu_stream=7, wait_event=Mock())
        self.event_factory = Mock(side_effect=Event)
        self.npu = NS(Event=self.event_factory, current_stream=lambda: self.stream)
        for context in (
            patch.object(torch, "npu", self.npu, create=True),
            patch.object(torch, "empty_like", cpu_empty_like),
            patch.object(runtime, "SelectedKVPlanner", Planner),
        ):
            context.start()
            self.addCleanup(context.stop)
        self.runner = NS(
            device=torch.device("cpu"),
            vllm_config=NS(scheduler_config=NS(max_num_batched_tokens=8192, max_num_seqs=8)),
        )
        self.manager = runtime.SelectedKVRuntime(
            self.runner, owner_by_layer={"l0": "l0", "l1": "l0", "l2": "l2"}, source=object()
        )

    def test_native_only_runtime_has_no_bound_selected_work(self):
        for metadata in (None, [], {}):
            self.assertIsNone(self.manager.bind_step(None, (), (), (), metadata))
        for name in ("source_step", "batch", "_compute_stream", "current_owner", "current_plan", "request_metadata"):
            self.assertIsNone(getattr(self.manager, name))
        self.assertEqual((self.manager.histories, self.manager.starts, self.manager.queries), ([], [], []))
        self.manager.selection_planner.build.assert_not_called()
        self.manager.selection_planner.load.assert_not_called()
        self.event_factory.assert_not_called()
        self.manager.close()
        self.manager.selection_planner.close_quiescent.assert_called_once_with()

    def make_batch(
        self, modes=(False, True, False, True), queries=None, histories=None, order=None, table_transform=None
    ):
        queries = queries or [128] * len(modes)
        histories = histories or [65536] * len(modes)
        ids = tuple(f"r{i}" for i in range(len(modes)))
        starts = np.r_[0, np.cumsum(queries)].tolist()
        requests = tuple(
            NS(request_id=name, history_tokens=h, query_tokens=q, selected=mode)
            for name, h, q, mode in zip(ids, histories, queries, modes)
        )
        step = Step(tuple(requests[i] for i in order) if order else requests)
        columns = max((h + q + 127) // 128 for h, q in zip(histories, queries))
        table = torch.arange(len(ids) * columns, dtype=torch.int32).view(len(ids), columns)
        metadata = NS(
            block_size=128,
            num_actual_tokens=starts[-1],
            block_table=table if table_transform is None else table_transform(table),
            selected_kv=None,
        )
        self.manager.bind_step(step, ids, histories, starts, {name: metadata for name in self.manager.owner_by_layer})
        return step, self.manager.batch, metadata

    def load(self, batch, metadata, layer="l0", topk=None, cache=None):
        topk = topk if topk is not None else torch.zeros(metadata.num_actual_tokens, 1, 2048, dtype=torch.int32)
        cache = cache if cache is not None else (torch.empty(8193, 128, 1, 656, dtype=torch.int8),)
        batch.load(layer, topk, cache, metadata)
        return topk, cache

    def finish(self, batch, metadata):
        for layer in self.manager.owner_by_layer:
            if layer not in self.manager._submitted_layers:
                self.load(batch, metadata, layer)
        self.manager.finish_step()

    def test_mixed_reordering_uses_original_rows_and_noncontiguous_query_ranges(self):
        step, batch, metadata = self.make_batch(
            queries=[2048, 128, 1024, 256],
            order=(3, 1, 0, 2),
            table_transform=lambda table: torch.nn.functional.pad(table, (0, 3))[:, :-3],
        )
        self.assertFalse(metadata.block_table.is_contiguous())
        original_table = metadata.block_table.clone()
        self.assertEqual(self.manager.request_metadata.tolist(), [[1, 65536, 2048, 128], [3, 65536, 3200, 256]])
        topk, cache = self.load(batch, metadata)
        call = self.manager.selection_planner.build.call_args
        self.assertIs(call.args[0], topk)
        self.assertEqual(call.args[1:], ([65536, 65536], [2048, 3200], [128, 256]))
        loaded = self.manager.selection_planner.load.call_args.args
        self.assertIs(loaded[2], step.source)
        self.assertIs(loaded[3], self.manager.request_metadata)
        self.assertIs(loaded[4], metadata.block_table)
        self.assertIs(loaded[5], cache[0])
        torch.testing.assert_close(metadata.block_table, original_table)

    def test_main_completion_handoff_follows_copy_for_owner_and_topk_follower(self):
        step, batch, metadata = self.make_batch()
        order = []
        self.manager.selection_planner.load.side_effect = lambda *_: order.append("main_copy")
        step.main_copy_submitted.side_effect = lambda name: order.append(name)
        self.load(batch, metadata, "l0")
        self.load(batch, metadata, "l1")
        self.assertEqual(order, ["main_copy", "l0", "main_copy", "l1"])

    def test_failed_main_copy_cannot_release_li_prefetch(self):
        step, batch, metadata = self.make_batch()
        self.manager.selection_planner.load.side_effect = RuntimeError("copy submission failed")
        with self.assertRaisesRegex(RuntimeError, "copy submission failed"):
            self.load(batch, metadata)
        step.main_copy_submitted.assert_not_called()
        self.assertEqual(self.manager._submitted_layers, set())
        self.stream.wait_event.assert_called_once_with(step.source_ready)
        self.event_factory.assert_not_called()
        self.assertEqual(self.manager.request_metadata_buffer.shape, (8, 4))

    def test_binding_freezes_cpu_input_views_once_before_publication(self):
        ids = ["r0", "r1"]
        histories = np.asarray([65536, 131072], dtype=np.int64)
        starts = np.asarray([0, 128, 384], dtype=np.int32)
        step = Step(
            (
                NS(request_id="r0", history_tokens=65536, query_tokens=128, selected=True),
                NS(request_id="r1", history_tokens=131072, query_tokens=256, selected=True),
            )
        )
        metadata = NS(
            block_size=128,
            num_actual_tokens=384,
            block_table=torch.zeros(2, 1026, dtype=torch.int32),
            selected_kv=None,
        )
        batch = self.manager.bind_step(
            step, ids, histories, starts, {name: metadata for name in self.manager.owner_by_layer}
        )
        ids[0] = "changed"
        histories[:] = 0
        starts[:] = 0
        self.assertEqual(self.manager.request_metadata.tolist(), [[0, 65536, 0, 128], [1, 131072, 128, 256]])
        self.load(batch, metadata)
        self.assertEqual(
            self.manager.selection_planner.build.call_args.args[1:], ([65536, 131072], [0, 128], [128, 256])
        )

    def test_metadata_copy_stays_on_compute_and_ready_wait_precedes_selected_copy(self):
        copies, order = [], []
        real_copy = torch.Tensor.copy_

        def copy(target, host, *args, **kwargs):
            copies.append((self.npu.current_stream(), kwargs.get("non_blocking")))
            return real_copy(target, host, *args, **kwargs)

        with patch.object(torch.Tensor, "copy_", copy):
            step, batch, metadata = self.make_batch()
        self.assertEqual(copies, [(self.stream, True)])
        self.stream.wait_event.assert_not_called()
        planner = self.manager.selection_planner

        def build(*args):
            order.append("union")
            return planner.outputs

        planner.build.side_effect = build
        self.stream.wait_event.side_effect = lambda event: order.append(("ready", event))
        planner.load.side_effect = lambda *args: order.append("selected_copy")
        self.load(batch, metadata)
        self.assertEqual(order, ["union", ("ready", step.source_ready), "selected_copy"])

    def test_owner_union_is_reused_and_each_layer_reads_its_source(self):
        step, batch, metadata = self.make_batch()
        topk, cache = self.load(batch, metadata)
        self.load(batch, metadata, "l1", topk, cache)
        self.load(batch, metadata, "l2", topk, cache)
        planner = self.manager.selection_planner
        self.assertEqual(planner.build.call_count, 2)
        self.assertEqual(planner.load.call_count, 3)
        self.assertEqual(planner.load.call_args_list[0].args[:2], planner.load.call_args_list[1].args[:2])
        self.assertEqual(planner.load.call_args_list[1].args[:2], planner.load.call_args_list[2].args[:2])
        self.assertEqual(self.stream.wait_event.call_count, 1)
        self.manager.finish_step()
        self.event_factory.assert_called_once_with()
        completion = step.completion
        self.assertTrue(completion.recorded)
        step.finish.assert_called_once_with(completion=completion)

    def test_shared_output_consumers_run_before_later_owner_overwrites_it(self):
        # Real CPU storage, with kernel submissions deferred until every layer
        # has enqueued. This checks stream order, not device kernel mathematics.
        planner_type = sys.modules[f"{PACKAGE}.prefill_selected_plan"].SelectedKVPlanner
        planner = planner_type.__new__(planner_type)
        planner.device, planner._closed = torch.device("cpu"), False
        planner.capacity, planner.query_capacity, planner.max_requests = 128, 8192, 8
        planner.outputs = [torch.empty(8, 128, dtype=torch.int32), torch.empty(8, 8, dtype=torch.int32)]
        planner.scratch, planner.pending = [], []
        launch_owner = ctypes.CFUNCTYPE(ctypes.c_int32)(lambda: 0)
        planner.fused_launch = ctypes.cast(launch_owner, ctypes.c_void_p).value
        planner.handle, planner.copy_launch = ctypes.c_void_p(1), 2
        queue, observed, order, sfa_reads = [], [], [], []

        def enqueue_union(topk, scratch, outputs, histories, starts, queries, *args):
            def run():
                outputs[0].fill_(int(topk[0, 0, 0]))
                outputs[1].fill_(len(histories))

            order.append("union")
            queue.append(run)

        def enqueue_copy(ids, status, *args):
            order.append("copy")
            queue.append(lambda: observed.append((ids.clone(), status.clone())))

        def enqueue_wait(event):
            order.append("wait")
            queue.append(lambda: event.query())

        def enqueue_sfa():
            order.append("sfa")
            queue.append(lambda: sfa_reads.append(int(observed[-1][0][0, 0])))

        def drain():
            while queue:
                if queue[0]() is False:
                    return
                queue.pop(0)

        planner.ops = NS(batch_plan=enqueue_union, inplace_copy=enqueue_copy)
        self.manager.selection_planner = planner
        step, batch, metadata = self.make_batch()
        step.source_ready.done = False
        self.stream.wait_event.side_effect = enqueue_wait
        first = torch.full((metadata.num_actual_tokens, 1, 2048), 7, dtype=torch.int32)
        second = torch.full_like(first, 9)
        cache = (torch.empty(8193, 128, 1, 656, dtype=torch.int8),)
        self.load(batch, metadata, "l0", first, cache)
        enqueue_sfa()
        old_plan = self.manager.current_plan
        self.load(batch, metadata, "l1", first, cache)
        enqueue_sfa()
        self.load(batch, metadata, "l2", second, cache)
        enqueue_sfa()
        self.assertEqual(old_plan[0].data_ptr(), self.manager.current_plan[0].data_ptr())
        self.assertEqual(old_plan[1].data_ptr(), self.manager.current_plan[1].data_ptr())
        self.assertEqual(order, ["union", "wait", "copy", "sfa", "copy", "sfa", "union", "copy", "sfa"])
        self.assertEqual(observed, [])
        self.assertEqual(len(planner.pending), 5)
        drain()
        self.assertEqual(observed, [])
        self.assertEqual(sfa_reads, [])
        step.source_ready.done = True
        drain()
        self.assertEqual(queue, [])
        self.assertEqual([int(ids[0, 0]) for ids, _ in observed], [7, 7, 9])
        self.assertEqual(sfa_reads, [7, 7, 9])
        for _, status in observed:
            torch.testing.assert_close(status, torch.full_like(status, 2))

    def test_first_source_wait_or_copy_failure_cannot_skip_next_wait(self):
        for failure in ("wait", "copy"):
            with self.subTest(failure=failure):
                self.manager = runtime.SelectedKVRuntime(self.runner, owner_by_layer={"l0": "l0"}, source=object())
                step, batch, metadata = self.make_batch(modes=(True,))
                planner = self.manager.selection_planner
                target = self.stream.wait_event if failure == "wait" else planner.load
                target.side_effect = RuntimeError(f"{failure} failed")
                try:
                    with self.assertRaisesRegex(RuntimeError, f"{failure} failed"):
                        self.load(batch, metadata)
                    self.assertEqual(self.manager._submitted_layers, set())
                    if failure == "wait":
                        planner.load.assert_not_called()
                    self.manager.finish_step(failed=True)
                    step.finish.assert_called_once_with(failed=True)
                    with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
                        self.manager.bind_step(None, (), (), (), {})
                    planner.release_quiescent.assert_not_called()
                finally:
                    target.side_effect = None

    def test_no_payload_allocation_or_query_gather_during_layer_load(self):
        _, batch, metadata = self.make_batch(modes=(True, True))
        topk = torch.zeros(metadata.num_actual_tokens, 1, 2048, dtype=torch.int32)
        cache = (torch.empty(8193, 128, 1, 656, dtype=torch.int8),)
        with (
            patch.object(torch, "index_select", side_effect=AssertionError("packing is forbidden")),
            patch.object(torch, "empty", side_effect=AssertionError("per-layer allocation is forbidden")),
        ):
            self.load(batch, metadata, topk=topk, cache=cache)

    def test_selected_boundaries_support_all_histories_and_queries(self):
        # Binding checks execution bounds after the source has frozen eligibility.
        for h in (16384, 16385, 20480, 65536, 65537, 65663, 131072, 262145, 524287, 524288):
            for q in (1, 125, 127, 128, 129, 256, 512, 1023, 1024):
                with self.subTest(h=h, q=q):
                    _, batch, metadata = self.make_batch(modes=(True,), queries=[q], histories=[h])
                    self.finish(batch, metadata)

    def test_frozen_selected_request_outside_execution_bounds_is_rejected(self):
        for h, q in ((16383, 128), (524289, 128)):
            with self.subTest(H=h, Q=q):
                self.manager = runtime.SelectedKVRuntime(
                    self.runner, owner_by_layer={"l0": "l0", "l1": "l0", "l2": "l2"}, source=object()
                )
                with self.assertRaisesRegex(ValueError, "exceeds the validated execution bounds"):
                    self.make_batch(modes=(True,), queries=[q], histories=[h])
                self.manager.selection_planner.build.assert_not_called()
                self.manager.selection_planner.load.assert_not_called()

    def test_large_selected_queries_bind_and_reach_the_existing_planner(self):
        for q in (1025, 2048, 4097, 7984):
            with self.subTest(Q=q):
                _, batch, metadata = self.make_batch(modes=(False, True), queries=[1, q], histories=[128, 524288])
                self.load(batch, metadata)
                args = self.manager.selection_planner.build.call_args.args
                self.assertEqual(args[1:], ([524288], [1], [q]))
                self.finish(batch, metadata)

    def test_follower_before_owner_and_duplicate_layer_are_rejected(self):
        _, batch, metadata = self.make_batch()
        with self.assertRaisesRegex(RuntimeError, "outside its current owner group"):
            self.load(batch, metadata, "l1")
        self.load(batch, metadata)
        with self.assertRaisesRegex(RuntimeError, "duplicated"):
            self.load(batch, metadata)

    def test_old_owner_follower_cannot_read_the_next_owners_output(self):
        _, batch, metadata = self.make_batch()
        self.load(batch, metadata, "l0")
        self.load(batch, metadata, "l2")
        planner = self.manager.selection_planner
        with self.assertRaisesRegex(RuntimeError, "outside its current owner group"):
            self.load(batch, metadata, "l1")
        self.assertEqual((planner.build.call_count, planner.load.call_count), (2, 2))
        self.assertNotIn("l1", self.manager._submitted_layers)

    def test_changed_stream_rejects_before_first_load_or_next_owner(self):
        for first_layer in (None, "l0"):
            with self.subTest(first_layer=first_layer):
                self.manager = runtime.SelectedKVRuntime(
                    self.runner, owner_by_layer={"l0": "l0", "l1": "l0", "l2": "l2"}, source=object()
                )
                step, batch, metadata = self.make_batch()
                if first_layer is not None:
                    self.load(batch, metadata, first_layer)
                planner = self.manager.selection_planner
                builds, copies = planner.build.call_count, planner.load.call_count
                with (
                    patch.object(self.stream, "npu_stream", 99),
                    self.assertRaisesRegex(RuntimeError, "compute stream"),
                ):
                    self.load(batch, metadata, "l0" if first_layer is None else "l2")
                self.assertEqual((planner.build.call_count, planner.load.call_count), (builds, copies))
                self.manager.finish_step(failed=True)
                step.finish.assert_called_once_with(failed=True)
                self.event_factory.assert_not_called()
                planner.release_quiescent.assert_not_called()

    def test_changed_finish_stream_keeps_source_failed_without_success_event(self):
        step, batch, metadata = self.make_batch()
        for layer in self.manager.owner_by_layer:
            self.load(batch, metadata, layer)
        with patch.object(self.stream, "npu_stream", 99), self.assertRaisesRegex(RuntimeError, "compute stream"):
            self.manager.finish_step()
        step.finish.assert_called_once_with(failed=True)
        self.event_factory.assert_not_called()
        self.assertTrue(self.manager._failed)
        with self.assertRaisesRegex(RuntimeError, "until process exit"):
            self.manager.close()
        self.manager.selection_planner.close_quiescent.assert_not_called()

    def test_new_step_with_same_request_ids_rejects_old_handle_by_identity(self):
        _, old_batch, old_metadata = self.make_batch()
        self.finish(old_batch, old_metadata)
        self.assertEqual(self.stream.wait_event.call_count, 1)
        self.stream.wait_event.reset_mock()
        _, new_batch, new_metadata = self.make_batch()
        self.stream.wait_event.assert_not_called()
        self.load(new_batch, new_metadata)
        self.stream.wait_event.assert_called_once_with(self.manager.source_step.source_ready)
        self.assertIsNot(old_batch, new_batch)
        self.assertNotEqual(old_batch, new_batch)
        self.assertIsNone(old_metadata.selected_kv)
        planner = self.manager.selection_planner
        planner.build.reset_mock()
        planner.load.reset_mock()
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            self.load(old_batch, new_metadata)
        planner.build.assert_not_called()
        planner.load.assert_not_called()

    def test_success_waits_real_event_and_native_source_retirement(self):
        step, batch, metadata = self.make_batch()
        self.finish(batch, metadata)
        completion = step.completion
        completion.done = False
        step.try_retire.return_value = False
        with self.assertRaisesRegex(RuntimeError, "live readers"):
            self.manager.bind_step(None, (), (), (), {"l0": metadata})
        self.assertEqual(completion.waits, 1)
        self.manager.selection_planner.release_quiescent.assert_not_called()
        step.try_retire.return_value = True
        self.manager.bind_step(None, (), (), (), {"l0": metadata})
        self.assertIsNone(metadata.selected_kv)
        self.assertIsNone(self.manager.current_plan)
        self.assertIsNone(self.manager.current_owner)
        self.assertIsNone(self.manager.source_step)
        self.assertIsNone(self.manager.batch)
        self.assertIsNone(self.manager._compute_stream)
        self.manager.selection_planner.release_quiescent.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError, "Stale"):
            self.load(batch, metadata)

    def test_failed_step_has_no_fabricated_completion_and_cannot_reuse(self):
        step, batch, metadata = self.make_batch()
        self.load(batch, metadata)
        self.manager.finish_step(failed=True)
        step.finish.assert_called_once_with(failed=True)
        self.event_factory.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            self.manager.bind_step(None, (), (), (), {})
        with self.assertRaisesRegex(RuntimeError, "until process exit"):
            self.manager.close()
        self.manager.selection_planner.close_quiescent.assert_not_called()

    def test_missing_layer_is_an_error_not_a_successful_forward(self):
        step, batch, metadata = self.make_batch()
        self.load(batch, metadata)
        with self.assertRaisesRegex(RuntimeError, "every layer"):
            self.manager.finish_step()
        self.event_factory.assert_not_called()
        step.finish.assert_called_once_with(failed=True)

    def test_completion_record_failure_preserves_source_and_workspace(self):
        step, batch, metadata = self.make_batch()
        for name in self.manager.owner_by_layer:
            self.load(batch, metadata, name)
        self.event_factory.side_effect = RuntimeError("event failed")
        with self.assertRaisesRegex(RuntimeError, "event failed"):
            self.manager.finish_step()
        self.assertIsNone(step.completion)
        self.assertTrue(self.manager._failed)
        step.finish.assert_called_once_with(failed=True)

    def test_source_finish_failure_cannot_reuse_workspace_or_hide_original_error(self):
        step, batch, metadata = self.make_batch()
        for name in self.manager.owner_by_layer:
            self.load(batch, metadata, name)
        step.finish.side_effect = [ValueError("invalid source finish"), RuntimeError("retention also failed")]
        with self.assertRaisesRegex(ValueError, "invalid source finish"):
            self.manager.finish_step()
        self.assertTrue(self.manager._failed)
        self.assertIsNone(step.completion)
        self.assertEqual(step.finish.call_args.kwargs, {"failed": True})
        with self.assertRaisesRegex(RuntimeError, "cannot be reused"):
            self.manager.bind_step(None, (), (), (), {})

    def bind_real_source(self):
        # Load both real modules, including their imports. Source construction
        # alone needs the SDK; these regressions exercise ownership retirement.
        package = "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"
        directory = ROOT / "vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store"
        with patch.dict(sys.modules):
            for name in ("selected_kv_address", "selected_kv_source"):
                spec = importlib.util.spec_from_file_location(f"{package}.{name}", directory / f"{name}.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
        source_type, step_type = module.SelectedKVSource, module.SelectedLoadStep
        old, batch, metadata = self.make_batch()
        source = source_type.__new__(source_type)
        source._publication_failed = False
        source._leased_infos = {"old": object()}
        source.components = tuple(self.manager.owner_by_layer)
        source.worker = NS(m_store=NS(batch_remove_lease=Mock(return_value=0)))
        real_step = step_type(
            source,
            old.requests,
            {name: old.source for name in source.components},
            old.source_ready,
            ["key"],
            connector_metadata=NS(),
        )
        source.step = self.manager.source_step = real_step
        real_step.native_complete(["key"])
        for name in self.manager.owner_by_layer:
            self.load(batch, metadata, name)
        return source, real_step

    def test_real_source_device_failure_blocks_connector_before_next_runtime_bind(self):
        source, real_step = self.bind_real_source()
        event = Mock()
        event.query.side_effect = RuntimeError("device completion failed")
        self.event_factory.side_effect = None
        self.event_factory.return_value = event
        with self.assertRaisesRegex(RuntimeError, "device completion failed"):
            self.manager.finish_step()
        self.assertTrue(real_step._failed)
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            source.begin_prepare()
        self.assertIn("old", source._leased_infos)
        source.worker.m_store.batch_remove_lease.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "until process exit"):
            self.manager.close()

    def test_real_unfinished_source_rejects_reuse_and_preserves_owners(self):
        source, real_step = self.bind_real_source()
        with self.assertRaisesRegex(RuntimeError, "did not finish successfully"):
            self.manager.bind_step(None, (), (), (), {})
        self.assertIs(self.manager.source_step, real_step)
        self.assertIn("old", source._leased_infos)
        self.manager.selection_planner.release_quiescent.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "did not finish successfully"):
            self.manager.close()
        source.worker.m_store.batch_remove_lease.assert_not_called()
        self.manager.selection_planner.close_quiescent.assert_not_called()

    def test_retirement_wait_failure_poison_is_shared_with_connector(self):
        source, real_step = self.bind_real_source()
        event = Mock()
        event.query.return_value = False
        self.event_factory.side_effect = None
        self.event_factory.return_value = event
        self.manager.finish_step()
        event.synchronize.side_effect = RuntimeError("device wait failed")
        with self.assertRaisesRegex(RuntimeError, "device wait failed"):
            self.manager.close()
        self.assertTrue(real_step._failed)
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            source.begin_prepare()
        self.assertIn("old", source._leased_infos)
        source.worker.m_store.batch_remove_lease.assert_not_called()
        self.manager.selection_planner.close_quiescent.assert_not_called()

    def test_binding_checks_actual_order_and_hq_before_metadata_publication(self):
        for kind in ("ids", "duplicate_ids", "duplicate_source_ids", "history", "starts", "no_selected", "large_q"):
            with self.subTest(kind=kind):
                manager = runtime.SelectedKVRuntime(self.runner, owner_by_layer={"l0": "l0"}, source=object())
                step = Step((NS(request_id="r", history_tokens=65536, query_tokens=128, selected=True),))
                ids, histories, starts = ("r",), (65536,), (0, 128)
                if kind == "ids":
                    ids = ("other",)
                elif kind == "duplicate_ids":
                    ids, histories, starts = ("r", "r"), (65536, 65536), (0, 128, 256)
                elif kind == "duplicate_source_ids":
                    step.requests = step.requests * 2
                elif kind == "history":
                    histories = (131072,)
                elif kind == "starts":
                    starts = (0, 256)
                elif kind == "no_selected":
                    step.requests[0].selected = False
                else:
                    step.requests[0].query_tokens = manager.query_capacity + 1
                    starts = (0, manager.query_capacity + 1)
                with self.assertRaises(ValueError):
                    manager.bind_step(step, ids, histories, starts, {})
                self.assertIs(manager.source_step, step)
                self.assertIsNone(manager.batch)
                manager.finish_step(failed=True)
                step.finish.assert_called_once_with(failed=True)

    def test_binding_uses_constructor_request_capacity_before_copy(self):
        schedule = self.runner.vllm_config.scheduler_config
        for tokens, requests in ((1, 4), (127, 256), (128, 4)):
            capacity = min(tokens, requests)
            schedule.max_num_batched_tokens, schedule.max_num_seqs = tokens, requests
            modes_to_check = [(True,) * capacity]
            if capacity > 1:
                modes_to_check.append(tuple(i % 2 == 0 for i in range(capacity)))
            for modes in modes_to_check:
                with self.subTest(tokens=tokens, requests=requests, selected=sum(modes)):
                    self.manager = runtime.SelectedKVRuntime(self.runner, owner_by_layer={"l0": "l0"}, source=object())
                    _, batch, metadata = self.make_batch(
                        modes=modes, queries=[1] * capacity, histories=[65536 if mode else 128 for mode in modes]
                    )
                    self.assertIsNotNone(batch)
                    self.assertIs(metadata.selected_kv, batch)
                    self.assertEqual(self.manager.max_requests, capacity)
                    self.assertEqual(self.manager.request_metadata.shape, (sum(modes), 4))

            self.manager = runtime.SelectedKVRuntime(self.runner, owner_by_layer={"l0": "l0"}, source=object())
            with (
                self.subTest(tokens=tokens, requests=requests, over_capacity=True),
                patch.object(torch.Tensor, "copy_", side_effect=AssertionError("Metadata copy must not start")) as copy,
                self.assertRaisesRegex(ValueError, "request sets differ"),
            ):
                self.make_batch(modes=(True,) * (capacity + 1), queries=[1] * (capacity + 1))
            copy.assert_not_called()
            self.assertIsNone(self.manager.batch)
            self.assertIsNone(self.manager.request_metadata)

    def test_bad_cache_geometry_is_rejected_without_enqueue(self):
        _, batch, metadata = self.make_batch()
        for cache in (
            (torch.empty(2, 128, 1, 672, dtype=torch.int8),),
            (torch.empty(2, 128, 1, 656, dtype=torch.bfloat16),),
            (torch.empty(256, 656, dtype=torch.int8),),
        ):
            with self.subTest(shape=cache[0].shape), self.assertRaisesRegex(ValueError, "packed C8"):
                self.load(batch, metadata, cache=cache)
        self.manager.selection_planner.build.assert_not_called()
        self.manager.selection_planner.load.assert_not_called()

    def test_closed_runtime_is_idempotent_and_cannot_bind(self):
        self.manager.close()
        self.manager.close()
        self.manager.selection_planner.close_quiescent.assert_called_once_with()
        self.assertIsNone(self.manager.request_metadata_buffer)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.manager.bind_step(None, (), (), (), {})


class WholeBatchSFATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "vllm_ascend/attention/sfa_v1.py"
        tree = ast.parse(path.read_text())
        block = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.With)
            and any(
                isinstance(item.context_expr, ast.Call)
                and isinstance(item.context_expr.func, ast.Name)
                and item.context_expr.func.id == "attention_transfer_window"
                for item in node.items
            )
        )
        arguments = (
            "self, has_local_query, attn_metadata, ql_nope, q_pe, kv_cache, topk_indices, "
            "actual_seq_lengths_query, actual_seq_lengths_key"
        )
        function = ast.parse(f"def run({arguments}):\n    pass\n").body[0]
        function.body = [block, ast.Return(value=ast.Constant(None))]
        cls.namespace = {}
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])), str(path), "exec"),
            cls.namespace,
        )
        cls.run_block = staticmethod(cls.namespace["run"])

    def test_selected_fill_precedes_one_original_sfa_with_identical_arguments(self):
        for enabled, has_queries in ((False, True), (True, True), (True, False)):
            with self.subTest(enabled=enabled, has_queries=has_queries):
                events = []

                @contextmanager
                def window(events=events):
                    events.append("native_window")
                    yield

                selected = (
                    NS(load=Mock(side_effect=lambda *_, events=events: events.append("selected_copy")))
                    if enabled
                    else None
                )
                attention = NS(selected_kv=selected)
                impl = NS(
                    layer_name="layer",
                    _execute_sparse_flash_attention_process=Mock(
                        side_effect=lambda *_, events=events: events.append("sfa")
                    ),
                )
                q, rope, main, topk, cuq, kvlen = [object() for _ in range(6)]
                cache = (main, object(), object())
                self.namespace["attention_transfer_window"] = window
                self.run_block(impl, has_queries, attention, q, rope, cache, topk, cuq, kvlen)
                if has_queries:
                    impl._execute_sparse_flash_attention_process.assert_called_once_with(
                        q, rope, cache, topk, attention, cuq, kvlen
                    )
                    if enabled:
                        selected.load.assert_called_once_with("layer", topk, (main,), attention)
                    self.assertEqual(events, ["native_window", *(["selected_copy"] if enabled else []), "sfa"])
                else:
                    impl._execute_sparse_flash_attention_process.assert_not_called()
                    selected.load.assert_not_called()


class WorkspaceTest(unittest.TestCase):
    def test_real_runtime_and_planner_tensor_storage_matches_startup_charge(self):
        # All torch allocations below are real CPU storages. Only the NPU
        # library/handle boundary and CPU pinning are replaced. This measures
        # tensor bytes, not NPU allocator rounding or reserved-memory peaks.
        planner_module = sys.modules[f"{PACKAGE}.prefill_selected_plan"]
        entry = ctypes.CFUNCTYPE(ctypes.c_int32)(lambda: 0)
        library = NS(
            launch_sparse_prefill_packed_inplace=entry,
            launch_sparse_prefill_fused_plan=entry,
            create_sparse_prefill_fixed_handle=Mock(return_value=0),
            destroy_sparse_prefill_fused_handle=Mock(return_value=0),
        )
        empty_like = torch.empty_like

        def unpinned(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return empty_like(*args, **kwargs)

        with (
            patch.object(ctypes, "CDLL", return_value=library),
            patch.object(torch.ops, "load_library"),
            patch.object(torch, "empty_like", unpinned),
            patch.object(torch, "npu", NS(device=lambda device: nullcontext()), create=True),
        ):
            bytes_by_geometry = {}
            for tokens, requests, owners in (
                (1, 4, 1),
                (127, 256, 1),
                (127, 8, 1),
                (128, 8, 2),
                (1024, 8, 2),
                (8192, 64, 3),
                (8192, 64, 12),
            ):
                with self.subTest(tokens=tokens, requests=requests, owners=owners):
                    runner = NS(
                        device=torch.device("cpu"),
                        vllm_config=NS(scheduler_config=NS(max_num_batched_tokens=tokens, max_num_seqs=requests)),
                    )
                    manager = runtime.SelectedKVRuntime(
                        runner, owner_by_layer={f"l{i}": f"l{i}" for i in range(owners)}, source=None
                    )
                    planner = manager.selection_planner
                    self.assertEqual(manager.max_requests, min(requests, tokens))
                    self.assertEqual(manager.request_metadata_buffer.shape, (manager.max_requests, 4))
                    self.assertEqual(manager.host_metadata.shape, manager.request_metadata_buffer.shape)
                    self.assertEqual(planner.max_requests, manager.max_requests)
                    tensors = [manager.request_metadata_buffer, *planner.scratch, *planner.outputs]
                    storages = {
                        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes() for tensor in tensors
                    }
                    charge = planner_module.selected_plan_storage_bytes(
                        workspace.selected_union_capacity(tokens),
                        tokens,
                        workspace.selected_request_capacity(tokens, min(requests, tokens)),
                    )
                    charge += workspace.selected_runtime_storage_bytes(tokens, min(requests, tokens))
                    self.assertEqual(sum(storages.values()), charge)
                    previous = bytes_by_geometry.setdefault((tokens, requests), charge)
                    self.assertEqual(previous, charge)
                    self.assertNotIn(manager.host_metadata.untyped_storage().data_ptr(), storages)
                    manager.close()

    def test_union_and_request_bounds_cover_every_eligible_request(self):
        for tokens in (1, 127, 128, 255, 256, 1024, 8192):
            for requests in (1, 8, 256):
                shape = workspace.selected_request_metadata_shape(tokens, requests)
                selected = min(requests, tokens)
                self.assertEqual(shape, (selected, 4))
                self.assertEqual(workspace.selected_runtime_storage_bytes(tokens, requests), selected * 16)
                for q in (1, 125, 127, 128, 129, 256, 512, 1023, 1024, 1025, 2048, 4097, 7984, 8192):
                    if q <= tokens:
                        for h in (65536, 131072, 262144, 524288):
                            self.assertLessEqual(min(h, q * 2048), workspace.selected_union_capacity(tokens))

    def test_invalid_workspace_is_rejected(self):
        for tokens, requests in ((0, 1), (8193, 1), (128, 257), (True, 1)):
            with self.assertRaises(ValueError):
                workspace.selected_request_metadata_shape(tokens, requests)


if __name__ == "__main__":
    unittest.main()
