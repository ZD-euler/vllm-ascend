# Selected main-KV prefill with native layerwise capacity

This opt-in implementation reduces historical main-KV transfers while retaining
native layerwise admission, logical block IDs, cache layouts and DDR storage.
Every request reserves its full native capacity. Selection fills the original
main-KV pages; it does not create a payload arena or change admission.
Union and source metadata scratch are charged before the original layerwise
budget factor, so enabling this feature can slightly reduce native capacity.

## Configuration and supported scope

Merge this into the existing Ascend additional configuration:

```json
{"selected_kv_config": {"enabled": true}}
```

The default is disabled. The implementation requires A3, eager model runner v1,
synchronous scheduling/loading, a PD producer with exactly the
`AscendStoreConnector` memcache GVA backend, and two original layerwise reuse
slots. Block-key layerwise loading is excluded. Supported geometry is
512 NoPE and 64 RoPE, TopK=2048, a 656-byte packed C8 main row, C8 LI
and 128-token pages. PP/PCP/DCP must be 1, and `enable_dsa_cp` must be false;
selected loading requires the original unsplit TopK rows. Speculation, DBO and sleep are
excluded. Automatic memory budgeting is required; explicit
`kv_cache_memory_bytes` is rejected when enabled. A block-count override cannot
bypass the physical workspace check. Query tensors and their dtype dispatch
remain with the original SFA; the selected kernels consume int32 TopK and
packed cache bytes, not query heads or query tensors.

The kernel supports at most 8192 scheduled query tokens and 256 requests per
worker. Within the current kernel domain, a request is eligible when H is
16K–512K, actual scheduled Q is positive, and Q/(H+Q) is strictly below
1.5%. The ratio uses the exact integer comparison `200 * Q < 3 * (H + Q)`;
a request at exactly 1.5% stays native. There is no separate per-request Q
cap or request-phase gate: nonfinal prefill chunks and post-prompt steps use
the same rule. The existing worker token budget and supported deployment
constraints still apply. H and Q need not be page-aligned. A partial history
page retains its original native full load;
selected copies skip it, preserving append into that same page.
Complete historical DDR sources must be leased and readable. Other requests
use the native path. The native model context limit may exceed this selected
history limit. There is no configurable payload partition.

## Data flow

1. Scheduler, `allocate_slots`, Coordinator and KV managers are unchanged.
   They allocate complete native logical page tables.
2. Worker-local binding reads each request's actual Q from the existing
   `SchedulerOutput.num_scheduled_tokens`. The source checks that connector
   metadata covers every request scheduled in this step; no Q or runtime object
   is added to the scheduler-to-worker transport metadata.
3. After native preparation, `SelectedKVSource` validates leased sources,
   all final request page references, actual per-layer load coverage and save
   sources. Repeated pages, NATIVE-referenced pages and historical pages used
   by saves retain full native main loading, as does a partial H/Q boundary
   page. A valid full-page locator is preserved; a missing boundary locator can
   use the leased native partial-page source only when its logical page matches.
   A selected page can be omitted
   only if it is exclusive, has a consistent leased source and is covered by
   the original prepared load. Save tasks remain unchanged.
4. The source publishes addresses for omitted pages only. Zero addresses mean
   proven native FULL loading, not missing sources. The address domain includes
   all selected history pages, including shared pages with high IDs. If no
   pages can be omitted, the whole step stays native without union work.
   Those same nonzero addresses also select complete LI data/scale pages for
   an eight-AIV MTE copy into the original LI cache. The receive thread admits
   it after the original previous-slot save and attention gates. For a
   prefetched LI layer, M3v13 also waits on a device event recorded after the
   triggering layer's selected main-copy submission. If that submission has
   not happened yet, the LI request is retained without blocking SDK transfers;
   whichever host callback arrives second enqueues the LI copy and its device
   wait. The consumer joins LI completion on its compute stream before append
   or indexer access. The first, unprefetched LI layer keeps its original gate.
   Shared, NATIVE, partial and append pages retain their SDK transfers. No LI
   index array, replacement cache, scheduler field or new HBM pool is added.
5. Runner binding checks final request order/H/Q and attaches an optional
   `AscendSFAMetadata.selected_kv` handle. A small tensor records original
   request row, H, original query start and Q once per step. This handle is
   execution metadata, not a persistent request tag or admission plan.
   The source owns the worker-local step, including its events and leases.
   Binding checks the prepared connector metadata by object identity; no
   runtime object is attached to the scheduler-to-worker metadata.
6. Original preprocessing, append and LI run normally. Before SFA, selected
   TopK ranges feed the existing local union/partition/finish algorithm.
   Nonadjacent requests use offsets into the original TopK tensor; local
   causal query numbering still starts at zero. Sorted historical IDs and
   count/error state stay on device; no compact remap is produced.
7. A device copy reads leased DDR rows directly into their original HBM
   positions. For historical token t, p = original_table[row, t // 128]:
   source = source_pages[p] + (t % 128) * 656;
   destination = native_main + (p * 128 + t % 128) * 656.
   Only omitted historical rows are copied. New Q stays in original append
   slots; native FULL pages are skipped. A main-copy event is recorded only
   when a prefetched LI layer depends on it. Recording an event does not mean
   H2D is complete: the LI stream waits for that actual device boundary.
8. The entire batch calls the original SFA once, with the original Q, TopK,
   page table and H+Q lengths. Projection, save and FFN stay in place.
   There is no wave, query/output packing, payload staging or second gather
   on device.

Source preflight scans page arrays once per shared task-data object, while
checking actual layer/component coverage for every layer. Source address
generation, clearing and uploading still cost O(layers × address domain).
Shared-prefix-heavy workloads can retain substantial full loading.

The LI dependency is worker-local step state. The original SDK task gates,
NATIVE copies, HBM/DDR layout, save path, scheduler and operator signatures
are unchanged. Deferred events and source leases live through the original
model-tail retirement; missing handoffs or submission failures cannot release
the generation. This later LI start trades less overlap with union/main-copy
for a shorter LI prefetch window. Measure next-layer LI waits and SFA as well
as forward before claiming a performance improvement over M3v12.

## Scratch and ownership

For token budget T and request bound R, selected metadata reserves
S = min(R, T) rows, since each selected request consumes at least one query token.
Union ID capacity per request is
U = min(524288, 2048 × T); every eligible union fits this upper bound.
A single ID/status output serves
the current contiguous TopK-owner group, costing 4 × S × (U + 8) bytes.
Intermediate groups and bitmap tables are also shared scratch. Once a later
owner starts, an earlier owner's follower is rejected before copy submission.
All selected loads in a step must use the same compute stream; a stream switch
is rejected before union or copy submission. Prior consumers therefore precede
the next union's overwrite on that stream. Startup accounting and actual
allocation use these same bounds, without a TopK-owner multiplier.
The local-union workspace reserves ceil(T/2) cohorts of 4096 int32 IDs and
eight int32 metadata entries. At T=8192, removing the former Q=1024 cap adds
56.109375 MiB of shared scratch, already included in startup accounting.
Partition processing reads that directory in fixed 512-record tiles, keeping
its UB footprint unchanged as Q increases.

Request metadata is one preallocated tensor with four int32 columns.
A new immutable batch handle identifies each
step, including repeated request IDs; no second numeric epoch is carried.
Source leases own the completion event used for retirement.

Runtime, source and step types are imported only for static checking at module
boundaries. Their optional handles are initialized explicitly before use;
completed work is retired through the existing events and failure retains its
owners. The native SDK's opaque key descriptors remain confined to the source
adapter. Worker-local objects never become transport metadata fields.

Local union reuses the MTP sort/extract primitives and retains both bounded
merge and low-overlap layouts. Each MrgSort source is limited to 4095 pairs.
A longer sorted run is split at 4092 pairs, keeping the second source aligned
to 32 bytes; the combined 8128-pair bound means at most one source needs this
split. One three-way merge replaces the two-way merge without extra scratch.
Partition/finish use hardware barriers and
publish sorted IDs plus device counts; no remap rank column is allocated.
The shared GM membership buffer uses 512K int32 entries (2 MiB), and the
bitmap uses 16K int32 entries (64 KiB).

Each device status record has eight int32 slots:
`[history_count, Q, history_count + Q, input_error, capacity_overflow, 0, 0, 0]`.
The copy consumer validates the count, Q and error flags on device; overflow
does not publish a truncated ID list or fall back to unfilled native pages.

Host launch, queued Torch operations and leased-address export share one
feature-local library. The three device build targets stay separate. This
library is loaded only for selected-KV use; it does not add a dependency to
the ordinary operator library.
The raw fixed-kernel launch adapter targets A3/CANN 8.5.1's generated device
object, symbol and hidden FFTS/overflow argument layout. This is not a portable
ABI promise across CANN releases: a different toolchain requires build and
device acceptance. Binary creation and explicit quiescent close use the
planner's device context; a failed explicit close retains the handle for retry.
If initialization and its rollback unload both fail, startup must terminate
the owning worker. In-process initialization retry is unsupported.

The original layer-load wait protects native FULL pages and LI. Source addresses
are published on a dedicated stream; union, selected copy and SFA run on the
current compute stream. The first selected copy joins the publication event;
later copies inherit that dependency through stream order.
Original post-SFA save events protect native slot reuse,
including empty saves and non-saving TP ranks. No extra slot waiter is installed.

Source leases and pinned metadata remain live until both real native completion
and a single step-end compute event complete. Every main layer must have
submitted before that event is recorded. Retirement is serialized under the
source's lock; device waits run outside that lock.
The original backend read lease has a finite TTL. Retaining Python owners does
not renew that lease or prove that DDR remains pinned after expiry. Deployment
acceptance must establish the installed SDK's expiry behavior and ensure that
failed or stalled workers cannot leave device readers alive beyond it.

Incomplete eligibility falls back before publishing any omission. Conflicting
page mappings and malformed metadata fail before publication. After publication,
binding/device/SFA errors retain source backing and scratch and stop reuse;
they do not fall back to unfilled native pages or invent completion events.
Transparent recovery after a failed selected step is outside this version.

## Validation boundary

CPU tests cover budgeting, final ordering, whole-batch SFA call identity,
shared/save/FULL coverage, native transfer builders, lease/error lifecycle and
host launcher contracts. Host simulations execute the actual local-union and
bitmap/finish headers against set-based expectations, including the 4095-pair
source limit, short/long query paths and high/medium/low overlap. Scalar
sort/merge models do not validate AscendC code generation, hardware fences or
device payload correctness. CPU-only review does not satisfy NPU acceptance.

CANN compilation and NPU acceptance remain required: exact 656-byte copy,
noncontiguous native pages, current-query preservation, shared/save guards,
poisoned unselected history rows, union edge cases (including the float
ReduceSum used for exact bounded popcounts), actual C8 SFA numerical
equality, failure propagation and warmed latency. Existing compact-path results
do not validate this in-place path. Measure preparation, union/finish, copy,
SFA, forward and HBM separately without adding overlapping or parallel ranks.

The selected device targets must preserve their fatal assertions in Release;
they cannot silently return on invalid device counts and let SFA continue.
[Ascend C's assertion documentation](https://asc.gitcode.com/api/Utils-API/tuning_interface/assert.html)
describes the disabling effect of NDEBUG and ASCENDC_DUMP. The target-local
build contract does not establish the installed SDK's macro expansion or
runtime error propagation: these require Release-build fault injection on A3.
