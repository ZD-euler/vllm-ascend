/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MemCache_Hybrid is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
*/
#ifndef MEM_FABRIC_MMC_CLIENT_DEFAULT_H
#define MEM_FABRIC_MMC_CLIENT_DEFAULT_H

#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>

#include "mmc_common_includes.h"
#include "mmc_meta_net_client.h"
#include "mmc_def.h"
#include "mmc_bm_proxy.h"
#include "mmc_interval_map.h"
#include "mmc_client_local_gva_blob_tracker.h"
#include "mmc_ubs_io_proxy.h"
#include "mmc_thread_pool.h"
#include "mmc_msg_client_meta.h"
#include "mmc_bandwidth_collector.h"

namespace ock {
namespace mmc {

enum ClientConfigField : uint32_t {
    CLIENT_CONFIG_FIELD_RPC_RETRY_TIMEOUT = 1U << 0,
    CLIENT_CONFIG_FIELD_AGGREGATE_IO = 1U << 1,
    CLIENT_CONFIG_FIELD_AGGREGATE_NUM = 1U << 2,
    CLIENT_CONFIG_FIELD_BATCH_CHUNK_SIZE = 1U << 3,
    CLIENT_CONFIG_FIELD_BATCH_CHUNK_COUNT = 1U << 4,
    CLIENT_CONFIG_FIELD_ALL = CLIENT_CONFIG_FIELD_RPC_RETRY_TIMEOUT | CLIENT_CONFIG_FIELD_AGGREGATE_IO |
                              CLIENT_CONFIG_FIELD_AGGREGATE_NUM | CLIENT_CONFIG_FIELD_BATCH_CHUNK_SIZE |
                              CLIENT_CONFIG_FIELD_BATCH_CHUNK_COUNT,
};

class MmcClientDefault : public MmcReferable {
public:
    ~MmcClientDefault() override = default;

    Result Start(const mmc_client_config_t &config);

    void Stop();

    // Hot-reload client runtime tunables (rpcRetryTimeOut, aggregate IO/num,
    // batch chunk size/count) from a freshly loaded mmc_client_config_t.
    // Thread-safe; only fields selected by fieldMask that differ from the cached
    // values are applied. Returns true when at least one field was applied.
    bool UpdateClientConfig(const mmc_client_config_t &config, uint32_t fieldMask = CLIENT_CONFIG_FIELD_ALL);

    const std::string &Name() const;

    Result Put(const char *key, mmc_buffer *buf, mmc_put_options &options, uint32_t flags);

    Result Get(const char *key, mmc_buffer *buf, uint32_t flags);

    Result Put(const std::string &key, const MmcBufferArray &bufArr, mmc_put_options &options, uint32_t flags);

    Result Get(const std::string &key, const MmcBufferArray &bufArr, uint32_t flags);

    Result BatchPut(const std::vector<std::string> &keys, const std::vector<mmc_buffer> &bufs, mmc_put_options &options,
                    uint32_t flags, std::vector<int> &batchResult);

    Result BatchGet(const std::vector<std::string> &keys, std::vector<mmc_buffer> &bufs, uint32_t flags,
                    std::vector<int> &batchResult);

    Result BatchPut(const std::vector<std::string> &keys, const std::vector<MmcBufferArray> &bufArrs,
                    mmc_put_options &options, uint32_t flags, std::vector<int> &batchResult);

    Result BatchGet(const std::vector<std::string> &keys, const std::vector<MmcBufferArray> &bufArrs, uint32_t flags,
                    std::vector<int> &batchResult);

    Result Remove(const char *key, uint32_t flags);

    Result BatchRemove(const std::vector<std::string> &keys, std::vector<Result> &remove_results, uint32_t flags);

    Result RemoveAll(uint32_t flags);

    Result IsExist(const std::string &key, uint32_t flags) const;

    Result BatchIsExist(const std::vector<std::string> &keys, std::vector<int32_t> &exist_results,
                        uint32_t flags) const;

    Result Query(const std::string &key, mmc_data_info &query_info, uint32_t flags);

    Result BatchQuery(const std::vector<std::string> &keys, std::vector<mmc_data_info> &query_infos, uint32_t flags);

    Result PrefetchKeys(std::vector<std::string> keys, MediaType srcMedia = MediaType::MEDIA_SSD,
                        MediaType dstMedia = MediaType::MEDIA_DRAM, uint32_t flags = 0);

    Result EvictKeys(std::vector<std::string> keys, MediaType srcMedia = MediaType::MEDIA_DRAM,
                     MediaType dstMedia = MediaType::MEDIA_SSD, uint32_t flags = 0);

    Result BatchAddLease(const std::vector<std::string> &keys, uint64_t leaseTtlMs, std::vector<int> &results);

    Result BatchAddLeaseWithInfo(const std::vector<std::string> &keys, uint64_t leaseTtlMs,
                                std::vector<int> &results, std::vector<mmc_data_info> &infos);

    Result BatchRemoveLease(const std::vector<std::string> &keys);

    Result BatchMalloc(const std::vector<std::string> &keys, const std::vector<size_t> &sizes,
                       const mmc_put_options &options, uint64_t leaseTtlMs, std::vector<uintptr_t> &gvas);

    Result BatchCopy(std::vector<void *> &gvas, std::vector<void *> &buffers, std::vector<size_t> &sizes,
                     const int32_t direct);

    Result BatchWriteFinish(const std::vector<std::string> &keys, const std::vector<int32_t> &writeResults,
                            std::vector<int32_t> &outResults);

    Result RegisterBuffer(uint64_t addr, uint64_t size);

    Result UnRegisterBuffer(uint64_t addr, uint64_t size);

    uint32_t RankId() const
    {
        return rankId_;
    }

    bool HasBm() const
    {
        return bmInitialized_;
    }

    static Result RegisterInstance()
    {
        std::lock_guard<std::mutex> lock(gClientHandlerMtx);
        if (gClientHandler != nullptr) {
            MMC_LOG_INFO("client handler already registered");
            return MMC_OK;
        }
        gClientHandler = new (std::nothrow) MmcClientDefault("mmc_client");
        if (gClientHandler == nullptr) {
            MMC_LOG_ERROR("alloc client handler failed");
            return MMC_ERROR;
        }
        gClientHandler->gvaBlobTracker_.SetName(gClientHandler->name_);
        return MMC_OK;
    }

    static Result UnregisterInstance()
    {
        std::lock_guard<std::mutex> lock(gClientHandlerMtx);
        if (gClientHandler == nullptr) {
            MMC_LOG_INFO("client handler already unregistered");
            return MMC_OK;
        }
        delete gClientHandler;
        gClientHandler = nullptr;
        return MMC_OK;
    }

    static MmcClientDefault *GetInstance()
    {
        std::lock_guard<std::mutex> lock(gClientHandlerMtx);
        return gClientHandler;
    }

    uint32_t GetRpcRetryTimeOut() const
    {
        return rpcRetryTimeOut_.load(std::memory_order_relaxed);
    }
    bool GetAggregateIO() const
    {
        return aggregateIO_.load(std::memory_order_relaxed);
    }
    size_t GetAggregateNum() const
    {
        return aggregateNum_.load(std::memory_order_relaxed);
    }
    uint64_t GetBatchChunkSize() const
    {
        return batchChunkSize_.load(std::memory_order_relaxed);
    }
    uint32_t GetBatchChunkCount() const
    {
        return batchChunkCount_.load(std::memory_order_relaxed);
    }

private:
    Result BatchAddLeaseImpl(const std::vector<std::string> &keys, uint64_t leaseTtlMs,
                             std::vector<int> &results, std::vector<mmc_data_info> *infos);

    explicit MmcClientDefault(const std::string &name) : name_(name) {}
    explicit MmcClientDefault(const MmcClientDefault &) = delete;
    MmcClientDefault &operator=(const MmcClientDefault &) = delete;

    inline uint32_t RankId(const affinity_policy &policy);
    Result PrepareAllocOpt(const uint64_t blobSize, const mmc_put_options &options, uint32_t flags,
                           AllocOptions &allocOpt);
    Result PrepareBlob(const MmcBufferArray &bufArr, const MmcMemBlobDesc &blob, MediaType &mediaType,
                       BatchCopyDesc &copyDesc, bool blobIsSrc);
    Result PrepareMultiBlobs(const MmcBufferArray &bufArr, const std::vector<MmcMemBlobDesc> &blobs,
                             MediaType &mediaType, BatchCopyDesc &copyDesc, bool blobIsSrc);
    Result PutData2Blobs(const std::vector<std::string> &keys, const std::vector<MmcBufferArray> &bufArrs,
                         const BatchAllocResponse &allocResponse, std::vector<int> &batchResult);
    void WaitFeatures(std::vector<std::tuple<uint32_t, uint32_t, std::future<int32_t>>> &futures,
                      std::vector<int> &batchResult);
    Result SyncUpdateState(BatchUpdateRequest &updateRequest, const std::string &opName);
    void AsyncUpdateState(BatchUpdateRequest &updateRequest, const std::string &opName);
    void SyncUpdateLease(BatchUpdateLeaseRequest &request);
    void AsyncUpdateLease(BatchUpdateLeaseRequest &request);
    std::future<int32_t> SubmitPutTask(BatchCopyDesc &copyDesc, MediaType mediaType, bool asyncExec);
    std::future<int32_t> SubmitGetTask(BatchCopyDesc &copyDesc, MediaType mediaType, bool asyncExec);
    Result BatchDataOperation(std::vector<void *> &gvas, std::vector<void *> &buffers, std::vector<size_t> &sizes,
                              int32_t direct);
    Result BatchCopyWritePath(std::vector<void *> &gvas, std::vector<void *> &buffers, std::vector<size_t> &sizes,
                              int32_t direct);
    Result BatchCopyReadPath(std::vector<void *> &gvas, std::vector<void *> &buffers, std::vector<size_t> &sizes,
                             int32_t direct);
    Result InitMetricReporting();
    void ReportMetrics();
    void ProcessExpiredReadLeases();
    Result ExecuteConcurrently(const std::vector<void *> &gvas, const std::vector<void *> &buffers,
                               const std::vector<size_t> &sizes, bool isPut, MediaType mediaType, size_t chunkSize);

    // UBS IO相关数据结构，保留供后续 SSD→DRAM 回暖使用
    struct UbsIoBatchGetData {
        const std::vector<std::string> &keys;
        const std::vector<MmcBufferArray> &bufArrs;
        std::vector<int> &batchResult;
        std::vector<std::string> &ubsIoKeys;
        std::vector<void *> &bufs;
        std::vector<std::string> &fallbackKeys;
        std::vector<mmc_buffer> &fallbackBuffers;
    };

    void ProcessUbsIoBatchGetWithHBM(UbsIoBatchGetData &data);

private:
    static std::mutex gClientHandlerMtx;
    static MmcClientDefault *gClientHandler;
    std::mutex mutex_;
    bool started_ = false;
    MetaNetClientPtr metaNetClient_;
    MmcBmProxyPtr bmProxy_;
    MmcUbsIoProxyPtr ubsIoProxy_;
    std::string name_;
    uint32_t rankId_{UINT32_MAX};
    bool bmInitialized_{false};
    std::atomic<uint32_t> rpcRetryTimeOut_{0};
    uint64_t defaultTtlMs_ = MMC_DATA_TTL_MS;
    MmcThreadPoolPtr threadPool_;
    MmcThreadPoolPtr readThreadPool_;
    std::atomic<bool> aggregateIO_{false};
    std::atomic<size_t> aggregateNum_{0};
    MmcThreadPoolPtr writeThreadPool_;
    std::atomic<uint64_t> batchChunkSize_{0};
    std::atomic<uint32_t> batchChunkCount_{0};
    LocalGvaBlobTracker gvaBlobTracker_{};
    BandwidthCollector *bandwidthCollector_ = nullptr;
};

uint32_t MmcClientDefault::RankId(const affinity_policy &policy)
{
    if (policy == NATIVE_AFFINITY) {
        return rankId_;
    } else {
        MMC_LOG_ERROR("affinity policy " << policy << " not supported");
        return 0;
    }
}
using MmcClientDefaultPtr = MmcRef<MmcClientDefault>;
} // namespace mmc
} // namespace ock

#endif // MEM_FABRIC_MMC_CLIENT_DEFAULT_H
