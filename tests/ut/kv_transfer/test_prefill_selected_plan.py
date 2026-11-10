# SPDX-License-Identifier: Apache-2.0
"""CPU bounds/call-contract checks; no AscendC or NPU execution is implied."""

import ctypes
import importlib.util
import math
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


def load_plan():
    path = Path(__file__).resolve().parents[3] / (
        "vllm_ascend/distributed/kv_transfer/sparse_kv_offload/prefill_selected_plan.py"
    )
    spec = importlib.util.spec_from_file_location("_selected_plan_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PLAN = load_plan()


class SelectedPlanBoundsTest(unittest.TestCase):
    def test_reservation_matches_each_allocated_buffer(self):
        for capacity in (65536, 131072, 524288):
            for tokens, requests in ((128, 1), (2048, 8), (8192, 64)):
                scratch, outputs = PLAN.selected_plan_shapes(capacity, tokens, requests)
                self.assertEqual(len(scratch), 6)
                self.assertEqual(outputs, ((requests, capacity), (requests, 8)))
                actual = sum(math.prod(shape) * 4 for shape in scratch)
                actual += sum(math.prod(shape) * 4 for shape in outputs)
                self.assertEqual(actual, PLAN.selected_plan_storage_bytes(capacity, tokens, requests))

    def test_transient_scratch_does_not_multiply_by_request(self):
        small, outputs = PLAN.selected_plan_shapes(524288, 8192, 1)
        large, _ = PLAN.selected_plan_shapes(524288, 8192, 64)
        self.assertEqual(small, large)
        self.assertEqual(small[2:], ((524288,), (32, 8), (32, 8), (16384,)))
        delta = PLAN.selected_plan_storage_bytes(524288, 8192, 2) - PLAN.selected_plan_storage_bytes(524288, 8192, 1)
        self.assertEqual(delta, sum(math.prod(shape) * 4 for shape in outputs))

    def test_shared_scratch_covers_largest_local_query(self):
        for tokens in (1, 64, 128, 256, 1024, 1025, 2048, 4097, 8192):
            scratch, _ = PLAN.selected_plan_shapes(524288, tokens, 64)
            self.assertEqual(scratch[0], ((tokens + 1) // 2, 4096))
            self.assertEqual(scratch[1], ((tokens + 1) // 2, 8))
            max_width = ((524288 + 4095) // 4096) * 128
            self.assertGreaterEqual(math.prod(scratch[2]), 32 * max_width)
            self.assertEqual(math.prod(scratch[2]), 32 * max_width)
            self.assertEqual(math.prod(scratch[-1]), 524288 // 32)

    def test_invalid_bounds_are_rejected_before_allocation(self):
        for arguments in (
            (0, 128, 1),
            (65537, 128, 1),
            (524416, 128, 1),
            (65536, 8193, 1),
            (65536, 0, 1),
            (65536, 128, 0),
            (65536, 128, 257),
            (True, 128, 1),
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                PLAN.selected_plan_shapes(*arguments)


class SelectedPlanCallTest(unittest.TestCase):
    def setUp(self):
        self.plan = PLAN.SelectedKVPlanner.__new__(PLAN.SelectedKVPlanner)
        self.plan.device = torch.device("cpu")
        self.plan.capacity, self.plan.query_capacity, self.plan.max_requests = 128, 1024, 4
        self.plan._closed = False
        self.plan.pending, self.plan.scratch = [], []
        self.plan.ops = SimpleNamespace(batch_plan=Mock(), inplace_copy=Mock())
        self.plan.outputs = [torch.empty(4, 128, dtype=torch.int32), torch.empty(4, 8, dtype=torch.int32)]
        self.launch_owner = ctypes.CFUNCTYPE(ctypes.c_int32)(lambda: 0)
        self.plan.fused_launch = ctypes.cast(self.launch_owner, ctypes.c_void_p).value
        self.plan.handle, self.plan.copy_launch = ctypes.c_void_p(1), 2
        self.topk = torch.zeros(1024, 1, 2048, dtype=torch.int32)

    def test_original_topk_and_nonadjacent_query_spans_are_forwarded(self):
        ids, status = self.plan.build(self.topk, [65536, 131072], [128, 768], [256, 128])
        args = self.plan.ops.batch_plan.call_args.args
        self.assertIs(args[0], self.topk)
        self.assertEqual(args[3:6], ([65536, 131072], [128, 768], [256, 128]))
        self.assertEqual((ids.shape, status.shape), ((2, 128), (2, 8)))
        self.assertEqual(ids.data_ptr(), self.plan.outputs[0].data_ptr())
        self.assertEqual(len(self.plan.pending), 1)
        self.assertEqual(args[-2], self.plan.fused_launch)
        self.assertIsInstance(args[-2], int)

    def test_configured_request_bounds_and_strict_types_reject_before_submission(self):
        for histories, starts, queries in (
            ([], [], []),
            ([65536] * 5, [0] * 5, [128] * 5),
            ([True], [0], [128]),
            ([65536], [0.0], [128]),
            ([65536], [0], [True]),
        ):
            with self.subTest(arguments=(histories, starts, queries)), self.assertRaises(ValueError):
                self.plan.build(self.topk, histories, starts, queries)
        self.plan.ops.batch_plan.assert_not_called()
        self.assertEqual(self.plan.pending, [])

    def test_configured_topk_rows_and_device_reject_before_submission(self):
        for topk in (
            torch.empty(0, 1, 2048, dtype=torch.int32),
            torch.empty(1025, 1, 2048, dtype=torch.int32),
            torch.empty(1024, 2048, dtype=torch.int32),
            torch.empty(1024, 1, 2048, dtype=torch.int32, device="meta"),
        ):
            with self.subTest(shape=topk.shape, device=topk.device), self.assertRaises(ValueError):
                self.plan.build(topk, [65536], [0], [128])
        self.plan.ops.batch_plan.assert_not_called()
        self.assertEqual(self.plan.pending, [])

    def test_host_plan_failure_preserves_runtimeerror_and_pending_owners(self):
        # Actual invalid spans are exercised by the real C++ wrapper host test.
        # Here only test propagation/ownership when that boundary raises.
        error = RuntimeError("host validation or enqueue failure")
        self.plan.ops.batch_plan.side_effect = error
        with self.assertRaises(RuntimeError) as raised:
            self.plan.build(self.topk, [65536], [0], [128])
        self.assertIs(raised.exception, error)
        self.assertEqual(len(self.plan.pending), 1)
        self.assertIs(self.plan.pending[0][0], self.topk)
        self.assertIs(self.plan.pending[0][1], self.plan.outputs)
        self.assertIs(self.plan.pending[0][2], self.plan.scratch)
        self.plan.release_quiescent()
        self.assertEqual(self.plan.pending, [])

    def copy_inputs(self):
        return [
            torch.zeros(2, 128, dtype=torch.int32),
            torch.zeros(2, 8, dtype=torch.int32),
            torch.ones(8, dtype=torch.uint64),
            torch.zeros(2, 4, dtype=torch.int32),
            torch.zeros(4, 12, dtype=torch.int32)[:, :8],
            torch.zeros(8, 128, 1, 656, dtype=torch.int8),
        ]

    def test_original_table_strided_view_is_not_copied(self):
        args = self.copy_inputs()
        self.assertFalse(args[4].is_contiguous())
        self.plan.load(*args)
        actual = self.plan.ops.inplace_copy.call_args.args
        self.assertTrue(all(a is b for a, b in zip(actual[:6], args)))
        self.assertEqual(actual[4].stride(), (12, 1))
        self.assertEqual(actual[-1], 2)

    def test_configured_copy_bounds_and_device_reject_before_submission(self):
        # Other invalid ABI cases moved to actual C++ pre-enqueue checks.
        replacements = (
            (0, torch.zeros(2, 127, dtype=torch.int32)),
            (0, torch.zeros(2, 256, dtype=torch.int32)),
            (0, torch.zeros(5, 128, dtype=torch.int32)),
            (0, torch.empty(2, 128, dtype=torch.int32, device="meta")),
        )
        for index, replacement in replacements:
            args = self.copy_inputs()
            args[index] = replacement
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.plan.load(*args)
        self.plan.ops.inplace_copy.assert_not_called()
        self.assertEqual(self.plan.pending, [])

    def test_failed_enqueue_keeps_tensor_owners_until_quiescence(self):
        args = self.copy_inputs()
        error = RuntimeError("enqueue failure")
        self.plan.ops.inplace_copy.side_effect = error
        with self.assertRaisesRegex(RuntimeError, "enqueue failure") as raised:
            self.plan.load(*args)
        self.assertIs(raised.exception, error)
        self.assertEqual(len(self.plan.pending), 1)
        self.assertTrue(all(a is b for a, b in zip(self.plan.pending[0], args)))
        self.plan.release_quiescent()
        self.assertEqual(self.plan.pending, [])

    def test_closed_plan_rejects_both_entries(self):
        self.plan._closed = True
        with self.assertRaises(RuntimeError):
            self.plan.build(self.topk, [65536], [0], [128])
        with self.assertRaises(RuntimeError):
            self.plan.load(*self.copy_inputs())


class SelectedPlanLifecycleTest(unittest.TestCase):
    """Only replace the ACL/device boundary; workspace allocations stay real."""

    def setUp(self):
        self.original_device = torch.device("privateuseone:0")
        self.target_device = torch.device("privateuseone:1")
        self.current_device = self.original_device
        self.calls = []
        self.create_status = self.destroy_status = 0
        self.fail_restore = False
        entry = ctypes.CFUNCTYPE(ctypes.c_int32)(lambda: 0)
        self.library = SimpleNamespace(
            launch_sparse_prefill_packed_inplace=entry,
            launch_sparse_prefill_fused_plan=entry,
            create_sparse_prefill_fixed_handle=Mock(side_effect=self.create),
            destroy_sparse_prefill_fused_handle=Mock(side_effect=self.destroy),
        )
        empty = torch.empty

        def allocate(shape, *, dtype, device):
            self.assertEqual(device, self.target_device)
            return empty(shape, dtype=dtype, device="cpu")

        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(torch, "npu", SimpleNamespace(device=self.device_context), create=True))
        stack.enter_context(patch.object(torch.ops, "load_library"))
        stack.enter_context(patch.object(ctypes, "CDLL", return_value=self.library))
        self.allocate = stack.enter_context(patch.object(torch, "empty", side_effect=allocate))

    @contextmanager
    def device_context(self, device):
        previous, self.current_device = self.current_device, device
        try:
            yield
        finally:
            self.current_device = previous
            if self.fail_restore:
                raise RuntimeError("device restore failed")

    def create(self, path, capacity, output):
        self.calls.append(("create", self.current_device))
        if not self.create_status:
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 0x1234
        return self.create_status

    def destroy(self, handle):
        self.calls.append(("destroy", self.current_device))
        self.assertEqual(handle.value, 0x1234)
        return self.destroy_status

    def make_plan(self):
        return PLAN.SelectedKVPlanner(self.target_device, 128, 128, 1)

    def test_create_and_close_use_the_planner_device_and_restore_the_caller(self):
        plan = self.make_plan()
        self.assertEqual(
            plan.fused_launch, ctypes.cast(plan.library.launch_sparse_prefill_fused_plan, ctypes.c_void_p).value
        )
        self.assertIsInstance(plan.fused_launch, int)
        self.assertEqual(self.current_device, self.original_device)
        plan.close_quiescent()
        self.assertEqual(self.calls, [("create", self.target_device), ("destroy", self.target_device)])
        self.assertEqual(self.current_device, self.original_device)
        self.assertIsNone(plan.handle.value)
        self.assertTrue(plan._closed)
        plan.close_quiescent()
        self.library.destroy_sparse_prefill_fused_handle.assert_called_once()

    def test_failed_destroy_keeps_handle_and_pending_owners_for_retry(self):
        plan = self.make_plan()
        owner = plan.outputs[0]
        plan.pending.append((owner,))
        self.destroy_status = 74
        with self.assertRaisesRegex(RuntimeError, "ACL status 74"):
            plan.close_quiescent()
        self.assertEqual(self.current_device, self.original_device)
        self.assertEqual(plan.handle.value, 0x1234)
        self.assertIs(plan.pending[0][0], owner)
        self.assertFalse(plan._closed)
        self.destroy_status = 0
        plan.close_quiescent()
        self.assertEqual(plan.pending, [])
        self.assertTrue(plan._closed)

    def test_successful_destroy_is_not_retried_if_device_restoration_fails(self):
        plan = self.make_plan()
        self.fail_restore = True
        with self.assertRaisesRegex(RuntimeError, "device restore failed"):
            plan.close_quiescent()
        self.assertTrue(plan._closed)
        self.assertIsNone(plan.handle.value)
        plan.close_quiescent()
        self.library.destroy_sparse_prefill_fused_handle.assert_called_once()

    def test_create_error_preserves_status_and_restores_the_caller_device(self):
        self.create_status = 72
        with self.assertRaisesRegex(RuntimeError, "ACL status 72"):
            self.make_plan()
        self.assertEqual(self.calls, [("create", self.target_device)])
        self.assertEqual(self.current_device, self.original_device)
        # The C++ create entry owns cleanup before publishing any live handle.
        self.library.destroy_sparse_prefill_fused_handle.assert_not_called()

    def test_allocation_failure_never_creates_an_acl_handle(self):
        self.allocate.side_effect = RuntimeError("workspace allocation failed")
        with self.assertRaisesRegex(RuntimeError, "workspace allocation failed"):
            self.make_plan()
        self.assertEqual(self.calls, [])
        self.library.create_sparse_prefill_fixed_handle.assert_not_called()


if __name__ == "__main__":
    unittest.main()
