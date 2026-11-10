# SPDX-License-Identifier: Apache-2.0
"""Selected main-KV reads over the unchanged native layerwise cache.

The connector owns DDR keys and the ordinary main/LI transfers. This module
suppresses selected historical main copies, loads their LI pages with MTE at
the native receive fence, and retains leases until all device readers finish.
It never changes native HBM addresses, append slots, save ranges, or page IDs.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from .selected_kv_address import HyBMAddressExporter

if TYPE_CHECKING:
    from .metadata import AscendConnectorMetadata, LayerTransferTask, ReqMeta, SharedBlockData
    from .pool_worker import KVPoolWorker

PAGE_ROWS = 128
MAIN_ROW_BYTES = 656
INDEXER_DATA_PAGE_BYTES = 128 * 128
INDEXER_SCALE_PAGE_BYTES = 128 * 2
MAX_HISTORY = 524288
MIN_HISTORY = 65536
MAX_QUERY = 1024


@dataclass(frozen=True)
class SelectedRequest:
    request_id: str
    history_tokens: int
    query_tokens: int
    selected: bool


def classify_request(request: ReqMeta, q: int) -> SelectedRequest:
    """Use the scheduled chunk, never rounded save or lookup lengths."""
    valid_q = type(q) is int and 0 < q <= request.target_token_len
    h = request.target_token_len - q if valid_q else request.target_token_len
    selected = (
        valid_q
        and MIN_HISTORY <= h <= MAX_HISTORY
        and q <= MAX_QUERY
        and request.is_last_chunk is True
        and request.num_prompt_tokens == request.target_token_len
    )
    return SelectedRequest(request.req_id, h, q if valid_q else 0, selected)


@dataclass(frozen=True)
class MainComponent:
    layer_name: str
    layer_id: int
    component: int
    object_offset: int
    page_bytes: int


@dataclass(frozen=True)
class IndexerComponent:
    """Actual native LI views and their offsets in the leased DDR object."""

    components: tuple[int, int]
    offsets: tuple[int, int]
    caches: tuple[torch.Tensor, torch.Tensor]


@dataclass(frozen=True, eq=False)
class MainLoadFilter:
    """Omit selected main and MTE-covered LI pages from native SDK batches."""

    keep_main: NDArray[np.bool_]
    components: tuple[int, ...]
    indexer_components: tuple[tuple[int, ...], ...] = ()

    def apply(
        self, layer_id: int, addresses: np.ndarray, sizes: np.ndarray, gvas: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        rows = self.keep_main.size
        if not rows:
            return addresses, sizes, gvas
        arrays = tuple(np.asarray(value).reshape(rows, -1) for value in (addresses, sizes, gvas))
        component = self.components[layer_id]
        if not 0 <= component < arrays[0].shape[1] or any(a.shape != arrays[0].shape for a in arrays):
            raise ValueError("Selected main filter does not match registered components")
        keep = np.ones(arrays[0].shape, dtype=np.bool_)
        keep[:, component] = self.keep_main
        if self.indexer_components:
            for index in self.indexer_components[layer_id]:
                keep[:, index] = self.keep_main
        return tuple(value[keep] for value in arrays)


class SelectedLoadStep:
    """One forward's selection and source lease, guarded by one lock.

    Native callbacks and model-thread retirement share this lifetime. Failed
    execution retains its keys and owners; completion is never manufactured.
    """

    def __init__(
        self,
        source: SelectedKVSource,
        requests: Sequence[SelectedRequest],
        page_bases: Mapping[str, torch.Tensor],
        ready: torch.npu.Event,
        keys: Iterable[str],
        *,
        connector_metadata: AscendConnectorMetadata,
        indexer_mte: bool = False,
    ) -> None:
        # Keep the prepared step's identity local; transport metadata remains data-only.
        self.connector_metadata = connector_metadata
        self.indexer_mte = indexer_mte
        self.requests = tuple(requests)
        self._source = source
        self._pages: Mapping[str, torch.Tensor] = MappingProxyType(dict(page_bases))
        self._ready = ready
        # The receive thread completes both SDK and selected LI MTE readers.
        self._keys = tuple(dict.fromkeys(keys))
        self._key_set = frozenset(self._keys)
        self._native_done: set[str] = set()
        self._completion: torch.npu.Event | None = None
        self._native_layers: set[int] = set()
        self._lock = threading.RLock()
        self._finished = self._failed = self._closed = False

    def page_bases(self, name: str) -> torch.Tensor:
        with self._lock:
            self._check_live(name)
            return self._pages[name]

    def ready(self, name: str) -> torch.npu.Event:
        with self._lock:
            self._check_live(name)
            return self._ready

    def _check_live(self, name: str | None = None) -> None:
        if self._failed or self._closed:
            raise RuntimeError("Selected KV source is failed or retired")
        if name is not None and name not in self._pages:
            raise ValueError("Selected KV layer does not belong to this forward")

    def native_complete(self, keys: Iterable[str]) -> None:
        with self._lock:
            # A genuine receive callback can arrive after a failed forward.
            # Record it without turning that failure into successful retirement.
            if self._closed:
                raise RuntimeError("Selected KV source is retired")
            keys = set(keys)
            if not keys.issubset(self._key_set):
                raise RuntimeError("Native completion contains another lease's keys")
            self._native_done.update(keys)

    def native_layer_ready(self, layer_id: int) -> None:
        with self._lock:
            self._check_live()
            if type(layer_id) is not int or not 0 <= layer_id < len(self._source.components):
                raise ValueError("Native completion does not belong to an actual layer")
            self._native_layers.add(layer_id)
            if len(self._native_layers) == len(self._source.components):
                # Each native wait returned successfully, including empty final
                # copies. This is completion evidence, not an invented DMA event.
                self._native_done.update(self._keys)

    def finish(self, completion: torch.npu.Event | None = None, failed: bool = False) -> bool:
        with self._lock:
            if self._finished:
                if failed and not self._closed:
                    # A caller can fail while querying completion or releasing
                    # its lease after successful submission. Never reopen it.
                    self._failed = True
                    return False
                raise RuntimeError("Selected forward has already finished")
            if not failed and (
                not callable(getattr(completion, "query", None))
                or not callable(getattr(completion, "synchronize", None))
            ):
                raise TypeError("Successful selected forward requires a real completion event")
            self._completion = None if failed else completion
            self._finished, self._failed = True, bool(failed)
        return self.try_retire()

    def try_retire(self) -> bool:
        with self._lock:
            if self._closed:
                return True
            if not self._finished or self._failed or self._native_done != self._key_set:
                return False
            try:
                complete = cast("torch.npu.Event", self._completion).query()
            except BaseException:
                self._failed = True
                raise
            if not complete:
                return False
            if self._keys:
                result = self._source.worker.m_store.batch_remove_lease(list(self._keys))
                if type(result) is not int or result != 0:
                    raise RuntimeError("Native read-lease release failed; ownership retained")
            self._closed = True
            return True

    def wait_completion(self) -> None:
        with self._lock:
            if not self._finished or self._failed:
                raise RuntimeError("Selected forward did not finish successfully; ownership retained")
            if self._closed:
                return
            event = cast("torch.npu.Event", self._completion)
        # Device waits must not hold the receive thread's lock.
        try:
            event.synchronize()
        except BaseException:
            with self._lock:
                self._failed = True
            raise


class SelectedKVSource:
    """Registered native component geometry plus at most one active forward."""

    def __init__(
        self,
        worker: KVPoolWorker,
        *,
        exporter: HyBMAddressExporter | None = None,
        event_factory: Callable[[], torch.npu.Event] | None = None,
        upload: Callable[[NDArray[np.uint64], torch.device], torch.Tensor] | None = None,
    ) -> None:
        if (
            not worker.use_layerwise_transfer
            or not worker.layerwise_offload
            or worker.use_block_key_layerwise
            or worker.load_async
            or worker.num_kv_cache_groups != 1
            or worker.grouped_block_size != [PAGE_ROWS]
            or worker.pp_size != 1
            or worker.dcp_size != 1
        ):
            raise ValueError("Selected main reads require synchronous native GVA layerwise geometry")
        self.worker = worker
        self.components = self._components()
        self.indexers = self._indexer_components()
        self.indexer_mte_enabled = True
        self.device = worker._as_cache_tuple(worker.kv_caches[self.components[0].layer_name])[0].device
        if exporter is None:
            sdk = sys.modules.get("memfabric_hybrid")
            sdk_path = getattr(sdk, "__file__", None)
            if not sdk_path:
                raise RuntimeError("Selected source requires the already initialized native SDK")
            library = Path(sdk_path).resolve(strict=True).parent / "lib/libmf_hybm_core.so"
            exporter = HyBMAddressExporter.from_loaded_library(library)
        self.exporter = exporter
        self._event = event_factory or torch.npu.Event
        self._upload = upload
        self._metadata_buffer: tuple[torch.Tensor, torch.Tensor] | None = None
        self._metadata_stream: torch.npu.Stream | None = None
        if upload is None:
            # Startup accounts for this exact capacity. Allocate before any
            # hook is installed, so the first selected step only slices/copies.
            capacity = len(self.components) * worker.num_blocks
            host = torch.empty(capacity, dtype=torch.uint64, pin_memory=True)
            self._metadata_stream = torch.npu.Stream(device=self.device)
            # Allocate on the first writer's stream, including allocator reuse.
            with torch.npu.stream(self._metadata_stream):
                device = torch.empty(capacity, dtype=torch.uint64, device=self.device)
            self._metadata_buffer = host, device
        # The native backend returns SDK-owned key descriptors without Python stubs.
        self._leased_infos: dict[str, Any] = {}
        self._prepared_metadata: AscendConnectorMetadata | None = None
        self.step: SelectedLoadStep | None = None
        self._publication_failed = False
        worker.kv_recv_thread.read_lease_release = self._release_native
        worker.kv_recv_thread.selected_indexer_load = self.load_indexer

    def _components(self) -> tuple[MainComponent, ...]:
        worker = self.worker
        reuse = worker._layerwise_reuse_layout
        if reuse is None or reuse.independent_layers or len(reuse.buffer_slots) != 2:
            raise ValueError("Selected main reads preserve exactly two native reuse slots")
        by_physical = reuse.layer_cache_specs
        order = sorted(by_physical)
        recv = worker.kv_recv_thread
        builder = recv.group_builders[0] if recv.group_builders else recv.layer_batch_builder
        result = []
        for layer_id, physical in enumerate(order):
            spec = by_physical[physical]
            name = spec.main.layer_name
            cache = worker._as_cache_tuple(worker.kv_caches[name])
            if (
                spec.extra_main_specs
                or len(cache) != 1
                or cache[0].dtype != torch.int8
                or tuple(cache[0].shape) != (worker.num_blocks, PAGE_ROWS, 1, MAIN_ROW_BYTES)
                or not cache[0].is_contiguous()
            ):
                raise ValueError("Selected main reads require the original packed C8 cache view")
            first, last = map(int, builder._layer_cache_entry_offsets_np[layer_id : layer_id + 2])
            # The explicit main layer name selected its registered tensor.
            # Length is checked below; it is never the component discriminator.
            matches = np.flatnonzero(builder._kv_caches_base_addr_np[first:last] == cache[0].data_ptr())
            if matches.size != 1:
                raise ValueError("Main cache identity does not uniquely match native registration")
            component = int(matches[0])
            index = first + component
            page_bytes = int(builder._block_len_np[index])
            if page_bytes != PAGE_ROWS * MAIN_ROW_BYTES or int(builder._block_stride_np[index]) != page_bytes:
                raise ValueError("Registered main page differs from packed C8 geometry")
            offset = int(builder.layer_byte_offset + builder._block_len_np[:index].sum())
            result.append(MainComponent(name, layer_id, component, offset, page_bytes))
        if len(result) != worker.num_layers:
            raise ValueError("Main component registry must cover every native model layer")
        return tuple(result)

    def _check_publication(self) -> None:
        if self._publication_failed:
            raise RuntimeError("Selected metadata publication failed; ownership retained")

    def _indexer_components(self) -> tuple[IndexerComponent | None, ...]:
        worker = self.worker
        recv = worker.kv_recv_thread
        builder = recv.group_builders[0] if recv.group_builders else recv.layer_batch_builder
        specs = worker._layerwise_reuse_layout.layer_cache_specs
        result = []
        for main, physical in zip(self.components, sorted(specs)):
            indexer = getattr(specs[physical], "indexer", None)
            if indexer is None:
                result.append(None)
                continue
            caches = worker._as_cache_tuple(worker.kv_caches[indexer.layer_name])
            lengths = (INDEXER_DATA_PAGE_BYTES, INDEXER_SCALE_PAGE_BYTES)
            dtypes = (torch.int8, torch.float16)
            if len(caches) != 2 or any(
                cache.dtype != dtype
                or not cache.is_contiguous()
                or cache.shape[0] != worker.num_blocks
                or cache.numel() * cache.element_size() != worker.num_blocks * length
                for cache, dtype, length in zip(caches, dtypes, lengths)
            ):
                raise ValueError("Selected LI copies require original C8 data and FP16 scale pages")
            first, last = map(int, builder._layer_cache_entry_offsets_np[main.layer_id : main.layer_id + 2])
            components, offsets = [], []
            for cache, length in zip(caches, lengths):
                matches = np.flatnonzero(builder._kv_caches_base_addr_np[first:last] == cache.data_ptr())
                if matches.size != 1:
                    raise ValueError("LI cache identity does not uniquely match native registration")
                component = int(matches[0])
                index = first + component
                if int(builder._block_len_np[index]) != length or int(builder._block_stride_np[index]) != length:
                    raise ValueError("Registered LI page differs from native C8 geometry")
                components.append(component)
                offsets.append(int(builder.layer_byte_offset + builder._block_len_np[:index].sum()))
            result.append(IndexerComponent(tuple(components), tuple(offsets), tuple(caches)))
        return tuple(result)

    def load_indexer(self, layer_id: int) -> None:
        """Replace selected LI copies at the original receive-thread fence.

        The caller has already waited for the previous slot's save and attention
        gate. Completion stays synchronous to that thread, so append, lease
        release and the original CPU load event keep their existing ordering.
        """
        step = self.step
        if step is None or not step.indexer_mte:
            return
        indexer = self.indexers[layer_id]
        if indexer is None:
            return
        main = self.components[layer_id]
        with torch.npu.stream(self._metadata_stream):
            torch.ops.dsa_prefill_queue.li_page_gather(
                step.page_bases(main.layer_name),
                *indexer.caches,
                main.object_offset,
                *indexer.offsets,
            )
            completion = self._event()
            completion.record()
        # A failed copy never publishes a native load-complete signal.
        completion.synchronize()

    def _release_native(self, keys: list[str]) -> None:
        self._check_publication()
        if self.step is None:
            self.worker.m_store.batch_remove_lease(keys)
        else:
            self.step.native_complete(keys)

    def native_layer_ready(self, layer_id: int) -> None:
        if self.step is not None:
            self.step.native_layer_ready(layer_id)

    def retire_previous(self) -> None:
        self._check_publication()
        if self.step is None:
            return
        self.step.wait_completion()
        if not self.step.try_retire():
            raise RuntimeError("Previous selected source has unfinished or failed consumers; ownership retained")
        self.step = None

    def begin_prepare(self) -> None:
        """Reset descriptors only after the preceding source is quiescent."""
        self._prepared_metadata = None
        self.retire_previous()
        self._leased_infos.clear()

    def native_prepared(self, metadata: AscendConnectorMetadata) -> None:
        """Publish the identity only after native task preparation succeeds."""
        self._check_publication()
        if self.step is not None or self._prepared_metadata is not None or self.worker.next_layer_to_submit != 0:
            raise RuntimeError("Native preparation must precede selected decisions and load submission")
        self._prepared_metadata = metadata

    def record_leased_infos(self, keys: Sequence[str], infos: Sequence[Any]) -> None:
        """Observe the existing native post-lease response; perform no RPC."""
        if self.step is not None or len(keys) != len(infos):
            raise ValueError("Leased descriptors must belong to native preparation")
        self._leased_infos.update(zip(keys, infos))

    def _publish(self, ids: np.ndarray, refs: NDArray[np.uint64], domain: int) -> torch.Tensor:
        shape = len(self.components), domain
        size = shape[0] * shape[1]
        capacity = len(self.components) * self.worker.num_blocks
        if size > capacity:
            raise ValueError("Selected source metadata exceeds its startup byte reservation")
        values: NDArray[np.uint64]
        if self._upload is not None:
            values = np.empty(shape, dtype=np.uint64)
        else:
            host, device = cast("tuple[torch.Tensor, torch.Tensor]", self._metadata_buffer)
            host, device = host[:size].view(shape), device[:size].view(shape)
            values = host.numpy()
        values.fill(0)
        for component in self.components:
            values[component.layer_id, ids] = refs + np.uint64(component.object_offset)
        if self._upload is not None:
            return self._upload(values, self.device)
        device.copy_(host, non_blocking=True)
        return device

    def _prepared_shared(
        self, layer_tasks: Sequence[Sequence[LayerTransferTask]]
    ) -> tuple[dict[int, SharedBlockData], list[frozenset[int]]] | None:
        """Check layer references once; scan each shared page array only once."""
        if len(layer_tasks) != len(self.components):
            return None
        shared_objects, layers = {}, []
        for component, tasks in zip(self.components, layer_tasks):
            objects = set()
            for task in tasks:
                shared = task.shared_block_data
                if shared is None:
                    return None
                if (
                    task.group_id != 0
                    or task.layer_idx_in_group != component.layer_id
                    or task.use_key_major_ranges
                    or shared.block_keys is not None
                    or shared.main_load_filter is not None
                ):
                    raise ValueError("Selected source requires unfiltered native GVA layer metadata")
                key = id(shared)
                if key not in shared_objects:
                    ids, gvas = np.asarray(shared.block_ids_arr), np.asarray(shared.block_gvas_arr)
                    if (
                        ids.ndim != 1
                        or gvas.ndim != 1
                        or ids.shape != gvas.shape
                        or ids.dtype.kind not in "iu"
                        or gvas.dtype.kind not in "iu"
                        or np.any(ids <= 0)
                        or np.any(ids >= self.worker.num_blocks)
                        or np.any(gvas <= 0)
                    ):
                        raise ValueError("Invalid prepared native page/GVA arrays")
                    shared_objects[key] = shared
                objects.add(key)
            layers.append(frozenset(objects))
        return shared_objects, layers

    def _coverage(
        self,
        ids: np.ndarray,
        gvas: np.ndarray,
        loads: tuple[dict[int, SharedBlockData], list[frozenset[int]]],
        saves: tuple[dict[int, SharedBlockData], list[frozenset[int]]],
    ) -> np.ndarray | None:
        """Prove native full-page readers before suppressing any main transfer."""
        load_objects, layer_objects = loads
        relevant = {}
        for key, shared in load_objects.items():
            take = np.isin(shared.block_ids_arr, ids)
            pages = shared.block_ids_arr[take]
            positions = np.searchsorted(ids, pages)
            if np.any(shared.block_gvas_arr[take] != gvas[positions]):
                raise ValueError("A shared original page has conflicting native load sources")
            relevant[key] = pages
        # Native layers usually share one object. Cache coverage by that exact
        # reference set, while still checking every actual layer has its task.
        checked = set()
        for keys in layer_objects:
            if keys in checked:
                continue
            if not keys:
                return None
            pages = np.unique(np.concatenate([relevant[key] for key in keys]))
            if not np.array_equal(pages, ids):
                return None
            checked.add(keys)
        saved = [shared.block_ids_arr for shared in saves[0].values()]
        return np.unique(np.concatenate(saved)) if saved else np.empty(0, dtype=np.int32)

    def prepare(
        self, metadata: AscendConnectorMetadata, scheduled_tokens: Mapping[str, int]
    ) -> SelectedLoadStep | None:
        self._check_publication()
        if self.step is not None or self.worker.next_layer_to_submit != 0:
            raise RuntimeError("Selected decisions must be frozen before native load submission")
        if self._prepared_metadata is not metadata:
            raise RuntimeError("Selected KV requires native preparation of this connector metadata")
        # Consume the successful preparation once, including native fallbacks.
        self._prepared_metadata = None
        requests = tuple(metadata.requests)
        if any(type(q) is not int or q < 0 for q in scheduled_tokens.values()):
            return None
        request_ids = {request.req_id for request in requests}
        if request_ids != {request_id for request_id, q in scheduled_tokens.items() if q > 0}:
            return None
        if len(request_ids) != len(requests):
            raise ValueError("Duplicate request IDs in one connector step")
        plans = tuple(classify_request(r, scheduled_tokens[r.req_id]) for r in requests)
        if not any(plan.selected for plan in plans):
            return None
        with self.worker._invalid_block_ids_lock:
            if self.worker._invalid_block_ids:
                return None
        sizes = {}
        for key, info in self._leased_infos.items():
            if info is not None and self.worker._step_lease_result.get(key) == 0:
                for gva in info.gva_list():
                    sizes[int(gva)] = int(info.size())
        selected_gvas, selected_ids, references, full_ids, accepted = [], [], [], [], []
        for request, plan in zip(requests, plans):
            groups = request.block_ids_by_group_np or request.block_ids_by_group
            table = np.asarray(groups[0]) if groups else np.empty(0, dtype=np.int32)
            required_pages = (request.target_token_len + PAGE_ROWS - 1) // PAGE_ROWS
            if table.ndim != 1 or table.dtype.kind not in "iu":
                raise ValueError("Invalid original native page table")
            if required_pages > table.size:
                return None
            table = table[:required_pages]
            if np.any(table <= 0) or np.any(table >= self.worker.num_blocks):
                raise ValueError("Invalid original native page ID")
            references.append(table)
            count = (plan.history_tokens + PAGE_ROWS - 1) // PAGE_ROWS
            locators = request.load_block_gvas_by_group_np
            gvas = np.asarray(locators[0]) if locators else np.empty(0, dtype=np.uint64)
            load = request.load_spec
            hit = 0
            if load is not None:
                hit = load.kvpool_store_skip_tokens
                if hit is None:
                    hit = load.kvpool_cached_tokens
            if (
                plan.selected
                and plan.history_tokens % PAGE_ROWS
                and gvas.ndim == 1
                and count <= gvas.size
                and gvas[count - 1] == 0
                and request.partial_load_gva_per_group
                and (hit - 1) // PAGE_ROWS == count - 1
            ):
                # A full prefix hit may already load the H/Q boundary as a full
                # page. Only fill a missing locator from the same partial page.
                gvas = gvas[:count].copy()
                gvas[-1] = request.partial_load_gva_per_group[0]
            complete = (
                plan.selected
                and load is not None
                and load.can_load
                and hit >= plan.history_tokens
                and request.load_gva_block_offset == 0
                and gvas.ndim == 1
                and gvas.dtype.kind in "iu"
                and count <= gvas.size
                and all(int(gva) in sizes for gva in gvas[:count])
            )
            if complete:
                selected_gvas.append(gvas[:count])
                selected_ids.append(table[:count])
                if plan.history_tokens % PAGE_ROWS:
                    # Append writes into this page. Keep its native full load;
                    # the selected copy sees a zero base and cannot race append.
                    full_ids.append(table[count - 1 : count])
            else:
                full_ids.append(table)
            accepted.append(replace(plan, selected=bool(complete)))
        if not selected_ids:
            return None
        ids, gvas = np.concatenate(selected_ids), np.concatenate(selected_gvas)
        order = np.argsort(ids, kind="stable")
        ids, gvas = ids[order], gvas[order]
        same = ids[1:] == ids[:-1]
        if np.any(same & (gvas[1:] != gvas[:-1])):
            raise ValueError("A shared original page has conflicting leased sources")
        first = np.r_[True, ~same]
        ids, gvas = ids[first], gvas[first]
        indexer_mte = self.indexer_mte_enabled and any(c is not None for c in self.indexers)
        required = max(c.object_offset + c.page_bytes for c in self.components)
        if indexer_mte:
            required = max(
                required,
                max(
                    offset + length
                    for indexer in self.indexers
                    if indexer is not None
                    for offset, length in zip(indexer.offsets, (INDEXER_DATA_PAGE_BYTES, INDEXER_SCALE_PAGE_BYTES))
                ),
            )
        if any(sizes[int(gva)] < required for gva in gvas):
            return None
        loads = self._prepared_shared(self.worker.layer_load_tasks)
        saves = self._prepared_shared(self.worker.layer_save_tasks)
        if loads is None or saves is None:
            return None
        saved = self._coverage(ids, gvas, loads, saves)
        if saved is None:
            return None
        referenced, counts = np.unique(np.concatenate(references), return_counts=True)
        kept = [referenced[counts > 1], saved, *full_ids]
        omitted = ~np.isin(ids, np.concatenate(kept))
        if not omitted.any():
            return None
        domain = int(ids.max()) + 1  # Includes high-ID pages kept on native load.
        ids, gvas = ids[omitted], gvas[omitted]
        components = tuple(c.component for c in self.components)
        indexer_components = tuple(c.components if c is not None else () for c in self.indexers) if indexer_mte else ()
        replacements: dict[int, SharedBlockData] = {}
        for object_id, shared in loads[0].items():
            keep = np.isin(shared.block_ids_arr, ids, invert=True)
            if keep.all():
                replacements[object_id] = shared
            else:
                keep.flags.writeable = False
                replacements[object_id] = replace(
                    shared, main_load_filter=MainLoadFilter(keep, components, indexer_components)
                )
        unique_gvas, inverse = np.unique(gvas, return_inverse=True)
        unique_gvas = np.ascontiguousarray(unique_gvas, dtype=np.uint64)
        actual_sizes = np.array([sizes[int(gva)] for gva in unique_gvas], dtype=np.uint64)
        # The selected readers use only this fixed prefix of every object.
        # The exporter still checks each allocation and both span endpoints.
        device_bases = self.exporter.export_component_arrays(unique_gvas, actual_sizes, 0, required)
        try:
            # The existing consumer wait joins this upload just before the
            # selected copy. CPU preparation and native filtering stay here.
            context = torch.npu.stream(self._metadata_stream) if self._metadata_stream is not None else nullcontext()
            with context:
                published = self._publish(ids, device_bases[inverse], domain)
                ready = self._event()
                ready.record()
            keys = (key for request in requests for key in request.load_keys)
            step = SelectedLoadStep(
                self,
                accepted,
                {c.layer_name: published[c.layer_id] for c in self.components},
                ready,
                keys,
                connector_metadata=metadata,
                indexer_mte=indexer_mte,
            )
            for tasks in self.worker.layer_load_tasks:
                for task in tasks:
                    task.shared_block_data = replacements[id(task.shared_block_data)]
            self.step = step
            return step
        except BaseException:
            # A copy may already be queued even when the Python call raises.
            # Never overwrite its pinned source or release its read leases.
            self._publication_failed = True
            raise

    def close(self) -> None:
        # Failed/unfinished asynchronous work is retained, never declared done.
        self.retire_previous()
        self.worker.kv_recv_thread.read_lease_release = None
        self.worker.kv_recv_thread.selected_indexer_load = None
        self._metadata_buffer = None
        self._metadata_stream = None
        self._leased_infos.clear()
        self._prepared_metadata = None
