"""Test-only byte oracle, drafted against selected-source API; requires review."""
from __future__ import annotations
import numpy as np
import torch

PAGE_BYTES = (16384, 256)


def selected_pages(source, layer_id):
    """Reconstruct replaced native descriptor rows without changing leases."""
    result = {}
    for task in source.worker.layer_load_tasks[layer_id]:
        shared = task.shared_block_data
        filt = shared.main_load_filter
        if filt is None:
            continue
        ids = np.asarray(shared.block_ids_arr).reshape(-1)
        gvas = np.asarray(shared.block_gvas_arr).reshape(-1)
        if len(ids) != len(gvas) or len(ids) != len(filt.keep_main):
            raise ValueError('native descriptors and mask disagree')
        for page, gva in zip(ids[~filt.keep_main], gvas[~filt.keep_main]):
            page, gva = int(page), int(gva)
            if page in result and result[page] != gva:
                raise ValueError('selected shared page has conflicting GVA')
            result[page] = gva
    return sorted(result.items())


def compare_one_layer(source, layer_id, chunk_pages=1024):
    """Compare MTE bytes against SDK overwrite after the original MTE fence.

    Destructive to selected destination pages only in an oracle burst, and only
    after their MTE bytes have been saved. A mismatch raises before attention.
    Does not export source bytes, addresses, page IDs, or tokens.
    """
    step = source.step
    if step is None or not step.indexer_mte:
        raise RuntimeError('MTE-selected source step is required')
    component = source.indexers[layer_id]
    if component is None:
        return None
    pairs = selected_pages(source, layer_id)
    if not pairs:
        raise RuntimeError('no actual selected history pages for LI oracle')
    caches = component.caches
    recv = source.worker.kv_recv_thread
    checks = [0, 0]
    mismatches = [0, 0]
    max_errors = [0, 0]
    oracle_stream = torch.npu.Stream()
    dependency = getattr(step, "_indexer_events", {}).get(layer_id)
    if dependency is not None:
        oracle_stream.wait_event(dependency)
    for start in range(0, len(pairs), chunk_pages):
        chunk = pairs[start:start + chunk_pages]
        ids = np.asarray([p for p, _ in chunk], dtype=np.int64)
        gvas = np.asarray([g for _, g in chunk], dtype=np.uint64)
        if any(ids.min() < 0 or ids.max() >= t.shape[0] for t in caches):
            raise ValueError('selected logical page outside LI cache')
        with torch.npu.stream(oracle_stream):
            ids_dev = torch.as_tensor(ids, dtype=torch.int64, device=caches[0].device)
            old = [t.view(torch.uint8).view(-1, n).index_select(0, ids_dev).clone()
                   for t, n in zip(caches, PAGE_BYTES)]
            clone_done = torch.npu.Event()
            clone_done.record()
        # The SDK may overwrite the same native LI destinations only after
        # all MTE bytes have been captured in the dedicated oracle stream.
        clone_done.synchronize()
        offsets = np.asarray(component.offsets, dtype=np.uint64)
        source_addresses = (gvas[:, None] + offsets).astype(np.int64).ravel()
        base = np.asarray([t.data_ptr() for t in caches], dtype=np.int64)
        sizes = np.asarray(PAGE_BYTES, dtype=np.int64)
        destinations = (base + ids[:, None] * sizes).ravel()
        lengths = np.broadcast_to(sizes, (len(ids), 2)).copy().ravel()
        rc = recv._batch_copy_with_limits(source_addresses, destinations, lengths, 1,
            recv.max_transfer_blocks, recv.max_transfer_bytes)
        if rc != 0:
            raise RuntimeError('SDK oracle copy failed')
        for i, (previous, tensor, page_bytes) in enumerate(zip(old, caches, PAGE_BYTES)):
            with torch.npu.stream(oracle_stream):
                actual = tensor.view(torch.uint8).view(-1, page_bytes).index_select(0, ids_dev)
                mismatch = torch.count_nonzero(previous != actual)
                max_error = (previous.to(torch.int16) - actual.to(torch.int16)).abs().max()
                compare_done = torch.npu.Event()
                compare_done.record()
            compare_done.synchronize()
            checks[i] += int(previous.numel())
            mismatches[i] += int(mismatch.item())
            max_errors[i] = max(max_errors[i], int(max_error.item()))
    if any(mismatches):
        raise AssertionError('MTE LI pages differ from original SDK')
    return dict(layer=layer_id, selected_pages=len(pairs), checked_bytes=checks,
                mismatch_bytes=mismatches, max_abs_byte_error=max_errors, passed=True)
