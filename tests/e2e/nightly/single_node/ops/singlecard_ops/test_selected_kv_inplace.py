# SPDX-License-Identifier: Apache-2.0
"""A3 union/copy/SFA acceptance, using HBM sources to isolate address correctness.

This does NOT validate DDR/HyBM address export, leases or transfer performance.
Run separately from latency tests after building the selected-KV libraries.
"""

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("Requires an Ascend NPU", allow_module_level=True)


@pytest.mark.parametrize("h,q", [(65536, 1), (65536, 128), (65537, 125), (65663, 127), (262144, 2048)])
@torch.inference_mode()
def test_original_pages_preserve_full_sfa_and_poisoned_unselected_rows(h, q):
    # Device-only imports follow the module's CPU collection guard.
    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_plan import SelectedKVPlanner
    from vllm_ascend.utils import enable_custom_op

    if get_ascend_device_type() != AscendDeviceType.A3:
        pytest.skip("A3 selected-KV kernel")
    enable_custom_op()
    device = torch.device("npu:0")
    requests, width = 3, 2048
    page_rows, row_bytes, heads = 128, 656, 16
    columns = (h + q + page_rows - 1) // page_rows
    generator = torch.Generator().manual_seed(31031)
    page_count = requests * columns + 2  # Two guard pages outside every request.
    table_cpu = torch.randperm(requests * columns, generator=generator).reshape(requests, columns).to(torch.int32) + 1
    # Both selected requests share one FULL page with the native request.
    table_cpu[0, 0] = table_cpu[2, 0] = table_cpu[1, 0]
    selected_rows = (0, 2)  # Native row 1 separates their original TopK ranges.
    shape = (page_count, page_rows, 1)
    nope = torch.randint(-8, 8, (*shape, 512), generator=generator, dtype=torch.int8)
    rope = torch.randn((*shape, 64), generator=generator).to(torch.bfloat16)
    scale = torch.rand((*shape, 4), generator=generator) * 0.05 + 0.05
    reference_cpu = torch.cat((nope, rope.view(torch.int8), scale.view(torch.int8)), dim=-1)
    reference = reference_cpu.to(device)
    source = reference.clone()
    destination_cpu = reference_cpu.clone()
    omitted = set()
    # A partial history page also holds append rows and stays FULL/native.
    for row in selected_rows:
        omitted.update(table_cpu[row, 1 : h // page_rows].tolist())
    destination_cpu[list(omitted)] = -1  # NaN scales/RoPE in every unfilled row.
    expected_cpu = destination_cpu.clone()
    destination = destination_cpu.to(device)
    page_bytes = page_rows * row_bytes
    bases = torch.tensor(
        [source.data_ptr() + page * page_bytes if page in omitted else 0 for page in range(page_count)],
        dtype=torch.uint64,
        device=device,
    )
    topk_cpu = torch.empty(requests * q, 1, width, dtype=torch.int32)
    for row in range(requests):
        for local_q in range(q):
            indices = (torch.arange(width, dtype=torch.int32) * 17 + local_q * 3) % h
            indices[-2], indices[-1] = h + local_q, -1
            topk_cpu[row * q + local_q, 0] = indices
    expected_ids = []
    for row in selected_rows:
        indices = topk_cpu[row * q : (row + 1) * q].flatten()
        ids = torch.unique(indices[(indices >= 0) & (indices < h)], sorted=True)
        expected_ids.append(ids)
        for token in ids.tolist():
            page = int(table_cpu[row, token // page_rows])
            if page in omitted:
                expected_cpu[page, token % page_rows] = reference_cpu[page, token % page_rows]
    topk = topk_cpu.to(device)
    # Use the original padded row stride for copy, without a device gather.
    padded_table = torch.zeros(requests, columns + 3, dtype=torch.int32, device=device)
    padded_table[:, :columns].copy_(table_cpu.to(device))
    original_table = padded_table[:, :columns]
    metadata = torch.tensor([[row, h, row * q, q] for row in selected_rows], dtype=torch.int32, device=device)
    planner = SelectedKVPlanner(device, 524288, requests * q, len(selected_rows))
    try:
        ids, status = planner.build(topk, [h, h], [0, 2 * q], [q, q])
        planner.load(ids, status, bases, metadata, original_table, destination)
        torch.npu.synchronize()
        statuses, actual_ids = status.cpu(), ids.cpu()
        for i, expected in enumerate(expected_ids):
            assert statuses[i, 0].item() == len(expected)
            assert statuses[i, 1].item() == q
            assert statuses[i, 3:5].tolist() == [0, 0]
            torch.testing.assert_close(actual_ids[i, : len(expected)], expected, atol=0, rtol=0)
        # Includes exact 656-byte writes, guards, FULL/shared pages, Q and all
        # omitted rows that TopK did not select. No 672-byte GM overrun is valid.
        torch.testing.assert_close(destination.cpu(), expected_cpu, atol=0, rtol=0)
        torch.testing.assert_close(source.cpu(), reference_cpu, atol=0, rtol=0)

        query = torch.randn((requests * q, heads, 576), generator=generator).to(torch.bfloat16).to(device)
        cumulative = torch.arange(1, requests + 1, dtype=torch.int32, device=device) * q
        lengths = torch.full((requests,), h + q, dtype=torch.int32, device=device)
        # Original SFA owns its normal dense table, not a compact selected table.
        sfa_table = table_cpu.to(device)

        def sfa(cache):
            result = torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention_vllm(
                query=query,
                key=cache,
                value=cache,
                sparse_indices=topk,
                scale_value=576**-0.5,
                sparse_block_size=1,
                block_table=sfa_table,
                actual_seq_lengths_query=cumulative,
                actual_seq_lengths_kv=lengths,
                layout_query="TND",
                layout_kv="PA_BSND",
                sparse_mode=3,
                attention_mode=2,
                quant_scale_repo_mode=1,
                tile_size=128,
                key_quant_mode=2,
                value_quant_mode=2,
                rope_head_dim=64,
                return_softmax_lse=False,
            )
            return result[0] if isinstance(result, tuple) else result

        full_output, selected_output = sfa(reference), sfa(destination)
        torch.npu.synchronize()
        assert torch.isfinite(selected_output.cpu()).all()
        torch.testing.assert_close(selected_output.cpu(), full_output.cpu(), atol=0, rtol=0)
    finally:
        # A failing synchronization must not be treated as quiescence.
        torch.npu.synchronize()
        planner.close_quiescent()
