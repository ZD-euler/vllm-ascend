// SPDX-License-Identifier: Apache-2.0
// Public ACL launch for the counted-union AscendC kernel containing SyncAll.
// Keep the launch ABI separate from the legacy generator's per-file AIV type.
#include <acl/acl.h>
#include <cstdint>
#include <new>

namespace {
constexpr uint32_t KERNEL_CORES = 32U;
constexpr uint32_t MAX_BATCHED_TOKENS = 8192U;
constexpr const char* KERNEL_NAME = "sparse_prefill_partition_finish_fixed_0";
} // namespace

extern "C" int32_t launch_sparse_prefill_adaptive_prefix(
    void*, void*, void*, void*, uint32_t, uint32_t);

struct SparsePrefillFusedHandle {
    aclrtBinHandle binary = nullptr;
    aclrtFuncHandle function = nullptr;
    void* ffts = nullptr;
    uint32_t maxCapacity = 0U;
};

extern "C" int32_t create_sparse_prefill_fixed_handle(const char* path, uint32_t maxCapacity, void** result)
{
    if (!path || !result || maxCapacity == 0U || maxCapacity > 524288U || maxCapacity % 128U != 0U) return -1;
    auto* handle = new (std::nothrow) SparsePrefillFusedHandle();
    if (!handle) return -1;
    auto status = aclrtBinaryLoadFromFile(path, nullptr, &handle->binary);
    if (status == ACL_SUCCESS) {
        status = aclrtBinaryGetFunction(handle->binary, KERNEL_NAME, &handle->function);
    }
    if (status == ACL_SUCCESS) status = aclrtGetHardwareSyncAddr(&handle->ffts);
    if (status != ACL_SUCCESS) {
        if (handle->binary) aclrtBinaryUnLoad(handle->binary);
        delete handle;
        return status;
    }
    handle->maxCapacity = maxCapacity;
    *result = handle;
    return ACL_SUCCESS;
}

extern "C" int32_t destroy_sparse_prefill_fused_handle(void* opaque)
{
    auto* handle = static_cast<SparsePrefillFusedHandle*>(opaque);
    if (!handle) return ACL_SUCCESS;
    const auto status = aclrtBinaryUnLoad(handle->binary);
    if (status != ACL_SUCCESS) return status;
    delete handle;
    return ACL_SUCCESS;
}

extern "C" int32_t launch_sparse_prefill_fused_plan(
    void* opaque, void* stream, void* topk, void* lists, void* descriptors, void* membership,
    void* counters, void* groups, void* bitmap, void* sources, void* lengths,
    uint32_t history, uint32_t query, uint32_t width, uint32_t capacity)
{
    if (!opaque || !stream || !topk || !lists || !descriptors || !membership || !counters ||
        !groups || !bitmap || !sources || !lengths || history > 524288U ||
        query == 0U || query > MAX_BATCHED_TOKENS || capacity == 0U || capacity > 524288U || capacity % 128U != 0U ||
        width < 128U || width > 16384U || width % 128U != 0U || width * 32U < history) return -1;
    auto* handle = static_cast<SparsePrefillFusedHandle*>(opaque);
    if (handle->maxCapacity != 0U && capacity > handle->maxCapacity) return -1;
    struct Arguments {
        void* ffts;
        void* lists; void* descriptors; void* membership;
        void* counters; void* groups; void* bitmap; void* sources; void* lengths;
        uint32_t history; uint32_t query; uint32_t width; uint32_t capacity;
        void* overflow;
    } arguments{handle->ffts, lists, descriptors, membership, counters, groups, bitmap, sources, lengths,
                history, query, width, capacity, nullptr};
    aclrtLaunchKernelAttr attribute{};
    attribute.id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attribute.value.schemMode = 1; // Every core must be admitted before SyncAll.
    aclrtLaunchKernelCfg config{&attribute, 1};
    const auto status = launch_sparse_prefill_adaptive_prefix(stream, topk, lists, descriptors, history, query);
    if (status != ACL_SUCCESS) return status;
    return aclrtLaunchKernelWithHostArgs(handle->function, KERNEL_CORES, stream, &config,
                                         &arguments, sizeof(arguments), nullptr, 0);
}
