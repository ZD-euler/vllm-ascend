# SPDX-License-Identifier: Apache-2.0
"""Exercise real LI handoffs with delayed CPU device queues, not wall sleeps."""

import threading
import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import test_selected_kv_source as base


class Stream:
    def __init__(self):
        self.tasks = []

    def submit(self, action):
        self.tasks.append(lambda: (action(), True)[1])

    def wait_event(self, event):
        if not event.recorded:
            raise AssertionError("A device wait must never target an unrecorded event")
        self.tasks.append(lambda: event.done)

    def run(self):
        while self.tasks and self.tasks[0]():
            self.tasks.pop(0)


class AfterMainCopyTest(unittest.TestCase):
    def setUp(self):
        self.selected, self.worker, self.metadata, self.removed, _ = base.fixture(indexers=True)
        self.step = self.selected.prepare(self.metadata, base.scheduled_tokens(self.metadata))
        self.compute, self.transfer = Stream(), Stream()
        self.selected._metadata_stream = self.transfer
        self.local = threading.local()
        self.local.stream = self.compute
        self.executed, self.submitted = [], []
        owner = self

        class Completion:
            def __init__(self):
                self.done = self.recorded = False

            def record(self):
                self.recorded = True
                owner.current_stream().submit(lambda: setattr(self, "done", True))

            def query(self):
                raise AssertionError("No CPU completion poll in a per-layer hook")

            def synchronize(self):
                raise AssertionError("No CPU device wait in a per-layer hook")

        @contextmanager
        def stream(value):
            previous = self.current_stream()
            self.local.stream = value
            try:
                yield
            finally:
                self.local.stream = previous

        def copy(pages, data, scale, *offsets):
            layer = next(i for i, geometry in enumerate(self.selected.indexers) if geometry.caches[0] is data)
            self.submitted.append(layer)
            self.assertIs(self.current_stream(), self.transfer)
            self.assertIs(pages, self.step.page_bases(self.selected.components[layer].layer_name))
            self.transfer.submit(lambda: self.executed.append(("LI", layer)))

        self.completion_type = Completion
        self.selected._event = Completion
        self.copy = Mock(side_effect=copy)
        fake = NS(
            npu=NS(stream=stream, current_stream=self.current_stream),
            ops=NS(dsa_prefill_queue=NS(li_page_gather=self.copy)),
        )
        context = patch.object(base.source, "torch", fake)
        context.start()
        self.addCleanup(context.stop)

    def current_stream(self):
        return getattr(self.local, "stream", self.compute)

    def main_submitted(self, layer):
        self.compute.submit(lambda: self.executed.append(("main", layer)))
        self.step.main_copy_submitted(self.selected.components[layer].layer_name)

    def test_receive_first_never_launches_li_before_actual_main_completion(self):
        self.step.defer_indexer(1, 0)
        self.selected.load_indexer(1)
        self.assertEqual(self.submitted, [])
        self.assertEqual(self.step._pending_indexer, {1})
        self.main_submitted(0)
        self.assertEqual(self.submitted, [1])
        self.assertFalse(self.step._main_load_events[0].done)
        self.transfer.run()
        self.assertEqual(self.executed, [])
        self.compute.run()
        self.transfer.run()
        self.assertEqual(self.executed, [("main", 0), ("LI", 1)])
        self.assertTrue(self.step._indexer_events[1].done)

    def test_main_first_still_requires_native_receive_slot_gate(self):
        self.step.defer_indexer(1, 0)
        self.main_submitted(0)
        self.compute.run()
        self.assertEqual(self.submitted, [])
        self.selected.load_indexer(1)
        self.transfer.run()
        self.assertEqual(self.executed, [("main", 0), ("LI", 1)])

    def test_first_layer_keeps_its_original_no_predecessor_load(self):
        self.selected.load_indexer(0)
        self.transfer.run()
        self.assertEqual(self.executed, [("LI", 0)])
        self.assertEqual(self.step._main_load_events, {})

    def test_compute_consumer_waits_for_both_main_and_li_device_completion(self):
        self.step.defer_indexer(1, 0)
        self.selected.load_indexer(1)
        self.main_submitted(0)
        self.selected.native_layer_ready(1)
        self.compute.submit(lambda: self.executed.append(("indexer", 1)))
        self.compute.run()
        self.assertEqual(self.executed, [("main", 0)])
        self.transfer.run()
        self.compute.run()
        self.assertEqual(self.executed, [("main", 0), ("LI", 1), ("indexer", 1)])

    def test_multiple_prefetches_share_one_main_event_without_unrecorded_waits(self):
        self.step.defer_indexer(1, 0)
        self.step.defer_indexer(2, 0)
        self.selected.load_indexer(2)
        self.selected.load_indexer(1)
        self.main_submitted(0)
        self.assertEqual(self.submitted, [1, 2])
        self.compute.run()
        self.transfer.run()
        self.assertEqual(self.executed, [("main", 0), ("LI", 1), ("LI", 2)])
        self.assertEqual(len(self.step._main_load_events), 1)

    def test_racing_main_and_receive_callbacks_submit_exactly_once(self):
        self.step.defer_indexer(1, 0)
        barrier = threading.Barrier(2)
        errors = []

        def receive():
            try:
                barrier.wait(timeout=2)
                self.selected.load_indexer(1)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=receive)
        thread.start()
        barrier.wait(timeout=2)
        self.main_submitted(0)
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(self.submitted, [1])
        self.compute.run()
        self.transfer.run()
        self.assertEqual(self.executed, [("main", 0), ("LI", 1)])

    def test_duplicate_requests_or_main_notifications_do_not_enqueue_twice(self):
        self.step.defer_indexer(1, 0)
        with self.assertRaises(ValueError):
            self.step.defer_indexer(1, 0)
        self.selected.load_indexer(1)
        with self.assertRaisesRegex(RuntimeError, "already requested"):
            self.selected.load_indexer(1)
        self.main_submitted(0)
        with self.assertRaisesRegex(RuntimeError, "already recorded"):
            self.main_submitted(0)
        self.assertEqual(self.submitted, [1])

    def test_backward_dependency_is_rejected_and_missing_main_cannot_pass_join(self):
        for target, after in ((0, 0), (1, 1), (1, 2)):
            with self.assertRaises(ValueError):
                self.step.defer_indexer(target, after)
        self.step.defer_indexer(1, 0)
        self.selected.load_indexer(1)
        with self.assertRaisesRegex(RuntimeError, "completion event is missing"):
            self.selected.native_layer_ready(1)
        self.step.finish(failed=True)
        self.assertFalse(self.step.try_retire())
        self.assertEqual(self.removed, [])

    def test_main_event_record_failure_retains_failed_generation(self):
        self.step.defer_indexer(1, 0)
        self.selected.load_indexer(1)
        self.selected._event = lambda: NS(record=Mock(side_effect=RuntimeError("record failed")))
        with self.assertRaisesRegex(RuntimeError, "record failed"):
            self.main_submitted(0)
        self.assertTrue(self.step._failed)
        self.assertIn(0, self.step._main_load_events)
        self.assertEqual(self.step._pending_indexer, {1})
        self.assertEqual(self.submitted, [])
        self.assertEqual(self.removed, [])

    def test_deferred_kernel_failure_prevents_reuse_and_lease_release(self):
        self.step.defer_indexer(1, 0)
        self.selected.load_indexer(1)
        self.copy.side_effect = RuntimeError("enqueue failed")
        with self.assertRaisesRegex(RuntimeError, "enqueue failed"):
            self.main_submitted(0)
        self.assertTrue(self.step._failed)
        self.assertIs(self.selected.step, self.step)
        with self.assertRaisesRegex(RuntimeError, "ownership retained"):
            self.selected.begin_prepare()
        self.assertEqual(self.removed, [])

    def test_no_indexer_or_no_selected_step_preserves_native_path(self):
        self.selected.indexers = tuple(None for _ in self.selected.indexers)
        self.step.defer_indexer(1, 0)
        self.selected.load_indexer(1)
        self.main_submitted(0)
        self.assertEqual(self.step._main_load_events, {})
        self.assertEqual(self.submitted, [])
        self.selected.step = None
        self.selected.load_indexer(1)

    def test_deferred_generation_retains_leases_until_actual_model_tail(self):
        self.selected.load_indexer(0)
        self.selected.native_layer_ready(0)
        self.step.defer_indexer(1, 0)
        self.selected.load_indexer(1)
        self.main_submitted(0)
        self.selected.native_layer_ready(1)
        self.step.defer_indexer(2, 1)
        self.selected.load_indexer(2)
        self.main_submitted(1)
        self.selected.native_layer_ready(2)
        tail = base.Event(False)
        self.compute.submit(lambda: setattr(tail, "done", True))
        self.step.native_complete([r.req_id for r in self.metadata.requests])
        self.assertFalse(self.step.finish(tail))
        for _ in range(2):
            self.transfer.run()
            self.compute.run()
            self.assertFalse(self.step.try_retire())
            self.assertEqual(self.removed, [])
        self.transfer.run()
        self.compute.run()
        self.assertTrue(self.step.try_retire())
        self.assertEqual(self.executed, [("LI", 0), ("main", 0), ("LI", 1), ("main", 1), ("LI", 2)])
        self.assertEqual(self.removed, [(self.metadata.requests[0].req_id,)])
        self.selected.begin_prepare()
        base.reset_native_tasks(self.worker, self.metadata.requests)
        infos = self.worker._step_keyinfo_cache
        self.selected.record_leased_infos(list(infos), list(infos.values()))
        self.selected.native_prepared(self.metadata)
        next_step = self.selected.prepare(self.metadata, base.scheduled_tokens(self.metadata))
        self.assertIsNot(next_step, self.step)
        self.assertEqual(next_step._indexer_after_main, {})
        self.assertEqual(next_step._main_load_events, {})
        self.assertEqual(next_step._pending_indexer, set())

    def test_real_worker_binds_exact_trigger_before_enqueue_without_changing_sdk_gate(self):
        gate = object()
        ns = {"LayerLoadTask": lambda **kw: NS(**kw), "get_attention_compute_start_gate": lambda: gate}
        cls = base.load_methods("pool_worker.py", "KVPoolWorker", {"_submit_ready_layer_loads"}, ns)
        worker = cls()
        worker.__dict__.update(vars(self.worker))
        worker.selected_kv_source = self.selected
        worker.prefetch_layer_map = {2: 0}
        worker.num_prefetch_layers = 2
        worker.current_layer = worker.next_layer_to_submit = 0
        queued = []

        def enqueue(data):
            if data.attention_start_gate is not None:
                self.assertEqual(self.step._indexer_after_main[data.layer_id], worker.current_layer)
            queued.append(data)

        worker.kv_recv_thread.add_request = enqueue
        worker._submit_ready_layer_loads()
        self.assertEqual(self.step._indexer_after_main, {1: 0})
        self.assertIsNone(queued[0].attention_start_gate)
        self.assertIs(queued[1].attention_start_gate, gate)
        worker.current_layer = 1
        worker._submit_ready_layer_loads()
        self.assertEqual(self.step._indexer_after_main, {1: 0, 2: 1})
        self.assertEqual(queued[2].wait_for_save_layer, 0)
        self.assertIs(queued[2].attention_start_gate, gate)

    def test_native_only_worker_does_not_add_events_or_li_dependencies(self):
        with patch.object(base.source, "torch", base.torch):
            selected, original, metadata, _, _ = base.fixture([base.request(q=4096)], indexers=True)
            self.assertIsNone(selected.prepare(metadata, base.scheduled_tokens(metadata)))
        gate = object()
        ns = {"LayerLoadTask": lambda **kw: NS(**kw), "get_attention_compute_start_gate": lambda: gate}
        cls = base.load_methods("pool_worker.py", "KVPoolWorker", {"_submit_ready_layer_loads"}, ns)
        worker = cls()
        worker.__dict__.update(vars(original))
        worker.selected_kv_source = selected
        worker.prefetch_layer_map = {}
        worker.num_prefetch_layers = 2
        worker.current_layer = worker.next_layer_to_submit = 0
        worker.kv_recv_thread.add_request = Mock()
        worker._submit_ready_layer_loads()
        self.assertEqual(worker.kv_recv_thread.add_request.call_count, 2)
        self.assertIsNone(selected.step)
        selected.load_indexer(1)
        self.assertEqual(self.submitted, [])

    def test_mixed_native_demand_keeps_original_sdk_rows_and_tables(self):
        requests = [base.request(h=65537, q=125), base.request("native", q=4096, first=1024)]
        with patch.object(base.source, "torch", base.torch):
            self.selected, self.worker, self.metadata, self.removed, _ = base.fixture(requests, indexers=True)
            self.step = self.selected.prepare(self.metadata, base.scheduled_tokens(self.metadata))
        self.selected._metadata_stream = self.transfer
        self.selected._event = self.completion_type
        shared = self.worker.layer_load_tasks[1][0].shared_block_data
        ids, gvas, keep = (
            value.copy() for value in (shared.block_ids_arr, shared.block_gvas_arr, shared.main_load_filter.keep_main)
        )
        self.assertEqual([r.selected for r in self.step.requests], [True, False])
        self.assertTrue(keep[ids >= 1024].all())
        self.step.defer_indexer(1, 0)
        self.selected.load_indexer(1)
        self.main_submitted(0)
        self.transfer.run()
        self.assertEqual(self.executed, [])
        self.compute.run()
        self.transfer.run()
        self.assertEqual(self.executed, [("main", 0), ("LI", 1)])
        self.assertIs(self.worker.layer_load_tasks[1][0].shared_block_data, shared)
        for before, after in zip(
            (ids, gvas, keep), (shared.block_ids_arr, shared.block_gvas_arr, shared.main_load_filter.keep_main)
        ):
            base.np.testing.assert_array_equal(before, after)

    def test_real_receiver_continues_native_sdk_while_li_waits_for_main_submission(self):
        ns = base.NativeIntegrationTest().native_namespace()
        cls = base.load_methods("kv_transfer.py", "KVCacheStoreLayerRecvingThread", {"_handle_request"}, ns)
        recv = cls()
        array = base.np.array([1])
        meta = ns["LayerBatchReqMeta"](
            req_ids=["native"],
            is_last_chunks=[False],
            load_keys=[],
            gvas_array=array,
            addr_array=array,
            size_array=array,
        )
        recv.group_builders = [NS(build_addrs=lambda *_: meta)]
        recv.layer_load_finished_events = [threading.Event() for _ in range(3)]
        recv.external_slot_release_waiter = None
        recv.selected_indexer_load = self.selected.load_indexer
        recv._batch_copy_with_limits = Mock(return_value=0)
        recv._stagger_h2d_submit = Mock()
        recv.max_transfer_blocks = recv.max_transfer_bytes = 0
        recv.final_layer_id = 2
        recv.request_queue = NS(task_done=Mock())
        recv.get_event = threading.Event()
        task = NS(use_key_major_ranges=False, group_id=0, layer_idx_in_group=1, shared_block_data=object())
        gate = NS(wait=Mock(return_value=True))
        self.step.defer_indexer(1, 0)
        recv._handle_request(NS(wait_for_save_layer=None, transfer_tasks=[task], layer_id=1, attention_start_gate=gate))
        gate.wait.assert_called_once()
        recv._batch_copy_with_limits.assert_called_once()
        self.assertTrue(recv.layer_load_finished_events[1].is_set())
        self.assertEqual(self.submitted, [])
        self.main_submitted(0)
        self.compute.run()
        self.transfer.run()
        self.assertEqual(self.executed, [("main", 0), ("LI", 1)])


if __name__ == "__main__":
    unittest.main()
