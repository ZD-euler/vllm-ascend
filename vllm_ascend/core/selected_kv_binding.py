# SPDX-License-Identifier: Apache-2.0
"""Startup byte accounting and forward binding; native allocation is untouched."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, cast

from vllm_ascend.core.selected_kv_config import (
    MAIN_ROW_BYTES,
    NATIVE_ALIGNMENT,
    PAGE_ROWS,
    reserve_workspace_bytes,
)

if TYPE_CHECKING:
    import torch
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.selected_kv_source import SelectedLoadStep
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_runtime import (
        SelectedKVBatch,
        SelectedKVRuntime,
    )
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner, PerLayerAttnMetadata
    from vllm_ascend.worker.worker import NPUWorker

SELECTED_KV_LORA_RANK = 512
SELECTED_ROPE_HEAD_DIM = 64
SELECTED_TOPK_WIDTH = 2048


def _native_geometry(runner: NPUModelRunner, specs: dict[str, KVCacheSpec]) -> tuple[tuple[str, ...], int, int]:
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
        build_layerwise_reuse_layout,
        get_layerwise_reuse_config,
    )

    layers = runner.model_config.get_num_layers(runner.parallel_config)
    reuse = build_layerwise_reuse_layout(
        specs, layers, get_layerwise_reuse_config(runner.vllm_config.kv_transfer_config)
    )
    if (
        not reuse.has_layer_reuse
        or reuse.independent_layers
        or len(reuse.buffer_slots) != 2
        or len(reuse.layer_cache_specs) != layers
    ):
        raise ValueError("Selected KV requires exactly two original native reuse slots and no independent layers")
    names: list[str] = []
    covered: set[str] = set()
    page_bytes, allocations = 0, 0
    for physical in sorted(reuse.layer_cache_specs):
        pair = reuse.layer_cache_specs[physical]
        if (
            pair.extra_main_specs
            or pair.main.spec.block_size != PAGE_ROWS
            or pair.main.spec.page_size_bytes != PAGE_ROWS * MAIN_ROW_BYTES
        ):
            raise ValueError("Selected KV requires the original packed C8 main page geometry")
        names.append(pair.main.layer_name)
        covered.add(pair.main.layer_name)
        if pair.indexer is not None:
            covered.add(pair.indexer.layer_name)
    if covered != set(specs):
        raise ValueError("Selected KV must account for every native main and indexer cache")
    for slot in reuse.buffer_slots:
        page_bytes += reuse.layer_cache_specs[slot[0]].main.spec.page_size_bytes
        allocations += 1
        indexers = [reuse.layer_cache_specs[layer].indexer for layer in slot]
        indexers = [entry.spec for entry in indexers if entry is not None]
        if not indexers or any(spec.block_size != PAGE_ROWS for spec in indexers):
            raise ValueError("Selected KV requires native layerwise indexer pages")
        # The original worker factor uses the first indexer in a slot. Restrict
        # this first version to equal extents so that exact factor stays valid.
        if len({spec.page_size_bytes for spec in indexers}) != 1:
            raise ValueError("Selected KV requires equal indexer page extents within each native slot")
        page_bytes += indexers[0].page_size_bytes
        allocations += 1
    return tuple(names), page_bytes, allocations * NATIVE_ALIGNMENT


def _topk_owners(runner: NPUModelRunner, names: Sequence[str]) -> dict[str, str]:
    context = runner.compilation_config.static_forward_context
    owners: dict[str, str] = {}
    previous: str | None = None
    for name in names:
        impl = context[name].impl
        if (
            getattr(impl, "kv_lora_rank", None) != SELECTED_KV_LORA_RANK
            or getattr(impl, "qk_rope_head_dim", None) != SELECTED_ROPE_HEAD_DIM
        ):
            raise ValueError("Selected KV kernels require kv_lora_rank=512 and qk_rope_head_dim=64")
        if not impl.skip_topk:
            indexer = getattr(impl, "indexer", None)
            if getattr(indexer, "topk_output_width", None) != SELECTED_TOPK_WIDTH:
                raise ValueError("Selected KV kernels require each actual TopK owner to output 2048 indices")
            previous = name
        if previous is None:
            raise ValueError("Selected KV followers require an earlier actual TopK owner")
        owners[name] = previous
    return owners


def reserve_selected_kv_workspace(worker: NPUWorker, available_bytes: int) -> int:
    """Called before the existing native layerwise logical-budget scaling."""
    runner = worker.model_runner
    config = runner.ascend_config.selected_kv_config
    if not config.enabled:
        return available_bytes
    config.validate_deployment(worker.vllm_config)
    # This library is currently compiled only for A3, not a generic C8 ABI.
    from vllm.v1.core.kv_cache_utils import get_kv_cache_groups

    from vllm_ascend.device.device_config import get_ascend_device_type
    from vllm_ascend.device.hardware import AscendDeviceType
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_plan import (
        selected_plan_storage_bytes,
    )
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_workspace import (
        selected_runtime_storage_bytes,
        selected_union_capacity,
    )

    if get_ascend_device_type() != AscendDeviceType.A3:
        raise ValueError("Selected KV kernels currently require an A3 build")
    if runner.ascend_config.enable_dsa_cp:
        raise ValueError("Selected KV requires whole-batch TopK and does not support DSA-CP token sharding")
    if (
        not runner.ascend_config.enable_sparse_sfa_c8
        or not runner.ascend_config.enable_sparse_li_c8
        or runner.ascend_config.enable_sparse_li_c4
        or runner.ascend_config.kvpp_config.size != 1
    ):
        raise ValueError("Selected KV requires native C8 main/indexer caches without KVPP")
    specs = worker.get_kv_cache_spec()
    groups = get_kv_cache_groups(worker.vllm_config, specs)
    if len(groups) != 1 or getattr(groups[0], "host_resident", False):
        raise ValueError("Selected KV preserves one original device KV cache group")
    names, page_bytes, padding = _native_geometry(runner, specs)
    _topk_owners(runner, names)
    schedule = worker.vllm_config.scheduler_config
    tokens, requests = schedule.max_num_batched_tokens, min(schedule.max_num_seqs, schedule.max_num_batched_tokens)
    fixed = selected_plan_storage_bytes(selected_union_capacity(tokens), tokens, requests)
    fixed += selected_runtime_storage_bytes(tokens, requests)
    budget = reserve_workspace_bytes(available_bytes, page_bytes, fixed, 8 * len(names), padding)
    override = getattr(worker.cache_config, "num_gpu_blocks_override", None)
    if override is not None:
        budget.validate(override)
    runner._selected_kv_budget = budget
    return budget.native_budget_bytes


def _unique_storage_bytes(
    kv_caches: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
) -> int:
    storages: dict[int, int] = {}
    for values in kv_caches.values():
        values = values if isinstance(values, (tuple, list)) else (values,)
        for tensor in values:
            storage = tensor.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
    return sum(storages.values())


def initialize_selected_kv(
    runner: NPUModelRunner,
    kv_cache_config: KVCacheConfig,
    kv_caches: Mapping[str, torch.Tensor | tuple[torch.Tensor, ...] | list[torch.Tensor]],
) -> None:
    """After native registration; create only source leases and scratch."""
    config = runner.ascend_config.selected_kv_config
    if not config.enabled:
        return
    from vllm.distributed.kv_transfer import get_kv_transfer_group

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.layerwise_cache_layout import (
        get_layerwise_kv_cache_specs,
    )
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.selected_kv_source import SelectedKVSource
    from vllm_ascend.distributed.kv_transfer.sparse_kv_offload.prefill_selected_runtime import SelectedKVRuntime

    budget = runner._selected_kv_budget
    if budget is None:
        raise RuntimeError("Selected KV workspace must be budgeted before native cache initialization")
    if runner.selected_kv_runtime is not None:
        raise RuntimeError("Selected KV runtime must be initialized exactly once")
    names, _, _ = _native_geometry(runner, get_layerwise_kv_cache_specs(kv_cache_config))
    owners = _topk_owners(runner, names)
    budget.validate(kv_cache_config.num_blocks, _unique_storage_bytes(kv_caches))
    worker = get_kv_transfer_group().connector_worker
    if worker is None or getattr(worker, "selected_kv_source", None) is not None:
        raise RuntimeError("Selected KV requires one registered native connector worker")
    source = SelectedKVSource(worker)
    try:
        runtime = SelectedKVRuntime(runner, owner_by_layer=owners, source=source)
    except BaseException as error:
        # No forward exists yet, so undo the source hooks installed above.
        try:
            source.close()
        except BaseException as cleanup_error:
            if hasattr(error, "add_note"):
                error.add_note(f"Selected KV source initialization cleanup also failed: {cleanup_error!r}")
        raise
    worker.selected_kv_source = source
    runner.selected_kv_runtime = runtime


def _finish_failed(runtime: SelectedKVRuntime, step: SelectedLoadStep, error: BaseException) -> None:
    try:
        if runtime.source_step is step:
            runtime.finish_step(failed=True)
        elif runtime.source.step is step:
            # Preparation can publish/filter a source before runtime binding.
            step.finish(failed=True)
    except BaseException as cleanup_error:
        if hasattr(error, "add_note"):
            error.add_note(f"Selected KV cleanup also failed: {cleanup_error!r}")


@contextmanager
def selected_kv_forward(
    runner: NPUModelRunner,
    scheduler_output: SchedulerOutput,
    attention_metadata: PerLayerAttnMetadata | None,
) -> Iterator[SelectedKVBatch | None]:
    """Bind only after native connector prepare, and preserve model errors."""
    runtime = cast("SelectedKVRuntime", runner.selected_kv_runtime)
    metadata = scheduler_output.kv_connector_metadata
    previous_step = runtime.source_step
    step = runtime.source.prepare(metadata, scheduler_output.num_scheduled_tokens) if metadata is not None else None
    if step is None:
        # Retire the previous selected readers before running a native batch.
        # Its request ordering and H/Q are irrelevant to selected execution.
        runtime.bind_step(None, (), (), (), attention_metadata)
        yield None
        return
    try:
        if step.connector_metadata is not metadata:
            # Main copies may already be filtered; never fall back with another step's tasks.
            raise RuntimeError("Selected KV source was prepared for different connector metadata")
        count = runner.input_batch.num_reqs
        request_ids = runner.input_batch.req_ids[:count]
        histories = runner.input_batch.num_computed_tokens_cpu[:count]
        # Runtime freezes these existing CPU slices once, without a device sync.
        starts = runner.query_start_loc.cpu[: count + 1]
        batch = runtime.bind_step(step, request_ids, histories, starts, attention_metadata)
    except BaseException as error:
        # Retain only this newly published source, even if binding failed before
        # acquiring it. Never finish a previous runtime batch on its behalf.
        if previous_step is not step:
            _finish_failed(runtime, step, error)
        raise
    try:
        yield batch
    except BaseException as error:
        _finish_failed(runtime, step, error)
        raise
    else:
        if runtime.source_step is not step:
            raise RuntimeError("Selected KV source changed during the bound forward")
        runtime.finish_step()
