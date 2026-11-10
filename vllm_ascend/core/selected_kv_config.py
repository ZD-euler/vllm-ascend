# SPDX-License-Identifier: Apache-2.0
"""Opt-in selected main-KV loading over the native cache allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import ConfigDict, StrictBool

from vllm_ascend.config_utils import config

if TYPE_CHECKING:
    from vllm.config import VllmConfig

PAGE_ROWS = 128
MAIN_ROW_BYTES = 656
MAX_REQUESTS = 256
MAX_BATCHED_TOKENS = 8192
NATIVE_ALIGNMENT = 2 * 1024 * 1024


@config(config=ConfigDict(frozen=True, extra="forbid"))
class SelectedKVConfig:
    """Disabled by default; capacity and page tables always remain native."""

    enabled: StrictBool = False

    def validate_deployment(self, vllm_config: VllmConfig) -> None:
        if not self.enabled:
            return
        transfer = vllm_config.kv_transfer_config
        extra = getattr(transfer, "kv_connector_extra_config", None) or {}
        if (
            getattr(transfer, "kv_role", None) != "kv_producer"
            or getattr(transfer, "kv_connector", None) != "AscendStoreConnector"
            or extra.get("backend") != "memcache"
            or extra.get("use_layerwise") is not True
            or extra.get("load_async", False) is not False
        ):
            raise ValueError("Selected KV requires a synchronous memcache layerwise AscendStore PD producer")
        schedule, model, parallel = (
            vllm_config.scheduler_config,
            vllm_config.model_config,
            vllm_config.parallel_config,
        )
        if (
            getattr(schedule, "async_scheduling", None) is not False
            or not model.enforce_eager
            or getattr(vllm_config, "use_v2_model_runner", False)
            or getattr(parallel, "enable_dbo", False)
        ):
            raise ValueError("Selected KV requires eager model runner v1 without async scheduling or DBO")
        if (
            vllm_config.speculative_config is not None
            or getattr(model, "is_encoder_decoder", False)
            or any(
                getattr(parallel, name, 1) != 1
                for name in ("pipeline_parallel_size", "prefill_context_parallel_size", "decode_context_parallel_size")
            )
        ):
            raise ValueError("Selected KV requires a decoder-only model with PP/PCP/DCP=1 and no speculation")
        if getattr(model, "enable_sleep_mode", False):
            raise ValueError("Selected KV workspace does not support sleep mode")
        if getattr(vllm_config.cache_config, "kv_cache_memory_bytes", None) is not None:
            raise ValueError(
                "Selected KV currently requires automatic KV memory budgeting; explicit bytes are unchanged"
            )
        tokens = schedule.max_num_batched_tokens
        requests = min(schedule.max_num_seqs, tokens)
        if not 0 < tokens <= MAX_BATCHED_TOKENS or not 0 < requests <= MAX_REQUESTS:
            raise ValueError("Selected KV supports at most 8192 batched tokens and 256 active requests")
        if model.max_model_len <= 0:
            raise ValueError("Selected KV requires a positive native model context length")


@dataclass(frozen=True)
class SelectedKVBudget:
    """A physical-byte reservation, never an alternative KV allocator."""

    available_bytes: int
    native_page_bytes: int
    native_padding_bytes: int
    fixed_bytes: int
    bytes_per_block: int
    maximum_blocks: int
    native_budget_bytes: int

    def validate(self, num_blocks: int, native_storage_bytes: int | None = None) -> None:
        if type(num_blocks) is not int or not 0 < num_blocks <= self.maximum_blocks:
            raise ValueError("Final native block count exceeds the selected-KV workspace reservation")
        native = (
            num_blocks * self.native_page_bytes + self.native_padding_bytes
            if native_storage_bytes is None
            else native_storage_bytes
        )
        if native + self.fixed_bytes + self.bytes_per_block * num_blocks > self.available_bytes:
            raise ValueError("Native KV storage plus selected-KV workspace exceeds the physical memory budget")


def reserve_workspace_bytes(
    available_bytes: int, native_page_bytes: int, fixed_bytes: int, bytes_per_block: int, native_padding_bytes: int = 0
) -> SelectedKVBudget:
    """Reserve against a conservative native N bound before its reuse factor.

    The ordinary core still derives N. Bounding metadata by the pre-reservation
    native N avoids a circular dependency without changing that core policy.
    """
    if (
        any(
            type(value) is not int or value < 0
            for value in (available_bytes, fixed_bytes, bytes_per_block, native_padding_bytes)
        )
        or type(native_page_bytes) is not int
        or native_page_bytes <= 0
    ):
        raise ValueError("Workspace budgets must be nonnegative integer bytes with a positive native page size")
    maximum_blocks = (available_bytes - native_padding_bytes) // native_page_bytes
    native_budget = available_bytes - native_padding_bytes - fixed_bytes - bytes_per_block * maximum_blocks
    if maximum_blocks <= 0 or native_budget < native_page_bytes:
        raise ValueError("Insufficient physical memory for native KV capacity and selected-KV workspace")
    return SelectedKVBudget(
        available_bytes,
        native_page_bytes,
        native_padding_bytes,
        fixed_bytes,
        bytes_per_block,
        maximum_blocks,
        native_budget,
    )
