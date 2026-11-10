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

#include "mmc_client_default.h"
#include "mmc_msg_client_meta.h"
#include "mmc_mem_obj_meta.h"
#include "mmc_bm_proxy.h"
#include "mmc_montotonic.h"
#include "mmc_periodic_task.h"
#include "mmc_ptracer.h"
#include "mmc_client_metric_manager.h"
#include "mmc_bandwidth_collector.h"

#include <algorithm>
#include <chrono>

namespace ock {
namespace mmc {
namespace {
uint64_t NowMs()
{
    return ock::dagger::Monotonic::TimeUs() / 1000ULL;
}

uint64_t ToLocalLeaseDeadlineMs(const MmcMemBlobDesc &blob)
{
    return blob.leaseTimeoutTtlMs_ == 0 ? 0 : (NowMs() + blob.leaseTimeoutTtlMs_);
}

std::vector<uint64_t> ToLocalLeaseDeadlinesMs(const std::vector<std::vector<MmcMemBlobDesc>> &blobs)
{
    const uint64_t nowMs = NowMs();
    std::vector<uint64_t> deadlines;
    deadlines.reserve(blobs.size());
    for (const auto &blobList : blobs) {
        const uint64_t leaseTimeoutTtlMs = blobList.empty() ? 0 : blobList[0].leaseTimeoutTtlMs_;
        deadlines.push_back(leaseTimeoutTtlMs == 0 ? 0 : (nowMs + leaseTimeoutTtlMs));
    }
    return deadlines;
}

Result CheckLeaseDeadline(const uint64_t localLeaseDeadlineMs)
{
    if (localLeaseDeadlineMs != 0 && NowMs() > localLeaseDeadlineMs) {
        return MMC_LEASE_EXPIRED;
    }
    return MMC_OK;
}

Result CheckLeaseDeadlines(const std::vector<uint64_t> &localLeaseDeadlineMsVec)
{
    for (const auto &localLeaseDeadlineMs : localLeaseDeadlineMsVec) {
        Result ret = CheckLeaseDeadline(localLeaseDeadlineMs);
        if (ret != MMC_OK) {
            return ret;
        }
    }
    return MMC_OK;
}
} // namespace

constexpr int CLIENT_THREAD_COUNT = 2;
constexpr uint32_t MMC_REGISTER_SET_MARK_BIT = 1U;
constexpr uint32_t MMC_REGISTER_SET_LEFT_MARK = 1U;
constexpr int32_t MMC_BATCH_TRANSPORT = 1U;
constexpr int32_t MMC_ASYNC_TRANSPORT = 2U;
constexpr uint32_t KEY_MAX_LENTH = 256U;
constexpr uint32_t GVA_LEASE_CLEANUP_INTERVAL_SECONDS = 1U;
constexpr uint32_t CLIENT_METRIC_REPORT_INTERVAL_SECONDS = 30U;

MmcClientDefault *MmcClientDefault::gClientHandler = nullptr;
std::mutex MmcClientDefault::gClientHandlerMtx;

Result MmcClientDefault::Start(const mmc_client_config_t &config)
{
    MMC_LOG_TRACE("Starting client " << name_);
    std::lock_guard<std::mutex> guard(mutex_);
    if (started_) {
        MMC_LOG_TRACE("MetaService " << name_ << " already started");
        return MMC_OK;
    }
    bmProxy_ = MmcBmProxyFactory::GetInstance("bmProxyDefault");
    MMC_ASSERT_LOG_AND_RETURN(bmProxy_ != nullptr, "bmProxy_ is nullptr", MMC_MALLOC_FAILED);
    rankId_ = bmProxy_->RankId();
    bmInitialized_ = bmProxy_->IsReady();

    threadPool_ = MmcMakeRef<MmcThreadPool>("client_pool", 1);
    MMC_ASSERT_LOG_AND_RETURN(threadPool_ != nullptr, "threadPool_ is nullptr", MMC_MALLOC_FAILED);
    MMC_RETURN_ERROR(threadPool_->Start(), "thread pool start failed");

    bool bindCpu = false;
    std::string protocol(config.dataOpType);
    if (protocol == "host_urma") {
        bindCpu = true;
    }
    readThreadPool_ = MmcMakeRef<MmcThreadPool>("read_pool", config.readThreadPoolNum);
    aggregateIO_ = config.aggregateIO;
    aggregateNum_ = static_cast<size_t>(config.aggregateNum);
    MMC_ASSERT_LOG_AND_RETURN(readThreadPool_ != nullptr, "readThreadPool_ is nullptr", MMC_MALLOC_FAILED);
    MMC_RETURN_ERROR(readThreadPool_->Start(bindCpu), "read thread pool start failed");

    writeThreadPool_ = MmcMakeRef<MmcThreadPool>("write_pool", config.writeThreadPoolNum);
    MMC_ASSERT_LOG_AND_RETURN(writeThreadPool_ != nullptr, "writeThreadPool_ is nullptr", MMC_MALLOC_FAILED);
    MMC_RETURN_ERROR(writeThreadPool_->Start(bindCpu), "write thread pool start failed");

    MMC_ASSERT_LOG_AND_RETURN(memchr(config.discoveryURL, '\0', DISCOVERY_URL_SIZE) != nullptr,
                              "config.discoveryURL possibly unterminated", MMC_INVALID_PARAM);
    auto tmpNetClient = MetaNetClientFactory::GetInstance(config.discoveryURL, "MetaClientCommon").Get();
    MMC_ASSERT_LOG_AND_RETURN(tmpNetClient != nullptr, "tmpNetClient is nullptr", MMC_NEW_OBJECT_FAILED);
    if (!tmpNetClient->Status()) {
        NetEngineOptions options;
        options.name = name_;
        options.threadCount = CLIENT_THREAD_COUNT;
        options.rankId = rankId_;
        options.startListener = false;
        options.tlsOption = config.tlsConfig;
        options.logLevel = config.logLevel;
        options.logFunc = config.logFunc;
        MMC_RETURN_ERROR(tmpNetClient->Start(options), "Failed to start net server of local service " << name_);
        MMC_RETURN_ERROR(tmpNetClient->Connect(config.discoveryURL, !bmInitialized_),
                         "Failed to connect net server of local service " << name_);
    }

    metaNetClient_ = tmpNetClient;
    rpcRetryTimeOut_ = config.rpcRetryTimeOut;
    batchChunkSize_ = config.batchChunkSize;
    batchChunkCount_ = config.batchChunkCount;
    MMC_RETURN_ERROR(RegisterPeriodicTask("client_lease_cleanup", GVA_LEASE_CLEANUP_INTERVAL_SECONDS,
                                          [this]() { ProcessExpiredReadLeases(); }),
                     "Failed to register client periodic task");

    MMC_RETURN_ERROR(InitMetricReporting(), "Failed to init metric reporting");

    started_ = true;
    return MMC_OK;
}

bool MmcClientDefault::UpdateClientConfig(const mmc_client_config_t &config, uint32_t fieldMask)
{
    bool updated = false;
    const uint32_t newRpcRetryTimeOut = config.rpcRetryTimeOut;
    const uint32_t oldRpcRetryTimeout = rpcRetryTimeOut_.load(std::memory_order_relaxed);
    if ((fieldMask & CLIENT_CONFIG_FIELD_RPC_RETRY_TIMEOUT) != 0 && oldRpcRetryTimeout != newRpcRetryTimeOut) {
        MMC_LOG_TRACE("client rpcRetryTimeOut hot-reloaded: " << oldRpcRetryTimeout << " to " << newRpcRetryTimeOut);
        rpcRetryTimeOut_.store(newRpcRetryTimeOut, std::memory_order_relaxed);
        updated = true;
    }
    if ((fieldMask & CLIENT_CONFIG_FIELD_AGGREGATE_IO) != 0 &&
        aggregateIO_.load(std::memory_order_relaxed) != config.aggregateIO) {
        const std::string oldAggregateIo = aggregateIO_.load(std::memory_order_relaxed) ? "true" : "false";
        const std::string newAggregateIo = config.aggregateIO ? "true" : "false";
        MMC_LOG_TRACE("client aggregateIO hot-reloaded: " << oldAggregateIo << " to " << newAggregateIo);
        aggregateIO_.store(config.aggregateIO, std::memory_order_relaxed);
        updated = true;
    }
    const size_t newAggregateNum = static_cast<size_t>(config.aggregateNum);
    const size_t oldAggregateNum = aggregateNum_.load(std::memory_order_relaxed);
    if ((fieldMask & CLIENT_CONFIG_FIELD_AGGREGATE_NUM) != 0 && oldAggregateNum != newAggregateNum) {
        MMC_LOG_TRACE("client aggregateNum hot-reloaded: " << oldAggregateNum << " to " << newAggregateNum);
        aggregateNum_.store(newAggregateNum, std::memory_order_relaxed);
        updated = true;
    }
    const uint64_t oldChunkSize = batchChunkSize_.load(std::memory_order_relaxed);
    if ((fieldMask & CLIENT_CONFIG_FIELD_BATCH_CHUNK_SIZE) != 0 && oldChunkSize != config.batchChunkSize) {
        MMC_LOG_TRACE("client batchChunkSize hot-reloaded: " << oldChunkSize << " to " << config.batchChunkSize);
        batchChunkSize_.store(config.batchChunkSize, std::memory_order_relaxed);
        updated = true;
    }
    const uint32_t oldChunkCount = batchChunkCount_.load(std::memory_order_relaxed);
    if ((fieldMask & CLIENT_CONFIG_FIELD_BATCH_CHUNK_COUNT) != 0 && oldChunkCount != config.batchChunkCount) {
        MMC_LOG_TRACE("client batchChunkCount hot-reloaded: " << oldChunkCount << " to " << config.batchChunkCount);
        batchChunkCount_.store(config.batchChunkCount, std::memory_order_relaxed);
        updated = true;
    }
    return updated;
}

void MmcClientDefault::Stop()
{
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!started_) {
            MMC_LOG_WARN("MmcClientDefault has not been started");
            return;
        }
        started_ = false;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    if (readThreadPool_ != nullptr) {
        readThreadPool_->Destroy();
    }
    if (writeThreadPool_ != nullptr) {
        writeThreadPool_->Destroy();
    }
    if (threadPool_ != nullptr) {
        threadPool_->Destroy();
    }
    if (metaNetClient_ != nullptr) {
        metaNetClient_->Stop();
        metaNetClient_ = nullptr;
        MMC_LOG_TRACE("MetaNetClient stopped.");
    }
    gvaBlobTracker_.Clear();
}

const std::string &MmcClientDefault::Name() const
{
    return name_;
}

Result MmcClientDefault::Put(const char *key, mmc_buffer *buf, mmc_put_options &options, uint32_t flags)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (buf == nullptr || key == nullptr || key[0] == '\0' || strnlen(key, KEY_MAX_LENTH + 1) == KEY_MAX_LENTH + 1) {
        MMC_LOG_ERROR("Invalid arguments, buf=" << buf << ", key=" << static_cast<const void *>(key));
        return MMC_ERROR;
    }

    MmcBufferArray buffArr{};
    buffArr.AddBuffer(*buf);
    return Put(key, buffArr, options, flags);
}

Result MmcClientDefault::PrepareAllocOpt(const uint64_t blobSize, const mmc_put_options &options, uint32_t flags,
                                         AllocOptions &allocOpt)
{
    allocOpt.blobSize_ = blobSize;
    allocOpt.numBlobs_ = std::max<uint16_t>(options.replicaNum, 1u);
    allocOpt.mediaType_ = MEDIA_NONE;
    allocOpt.flags_ = flags;

    std::copy_if(std::begin(options.preferredLocalServiceIDs), std::end(options.preferredLocalServiceIDs),
                 std::back_inserter(allocOpt.preferredRank_), [](const int32_t x) { return x >= 0; });
    // preferredRank_数量需要小于等于replicaNum， 具体参考 MmcGlobalAllocator Alloc方法说明
    if (allocOpt.preferredRank_.size() > allocOpt.numBlobs_ || allocOpt.numBlobs_ > MAX_BLOB_COPIES) {
        MMC_LOG_ERROR("preferredRank size:" << allocOpt.preferredRank_.size()
                                            << " is greater than replicaNum:" << allocOpt.numBlobs_);
        return MMC_INVALID_PARAM;
    }

    if (!allocOpt.preferredRank_.empty()) {
        allocOpt.flags_ = allocOpt.flags_ & ~0xFF; // 清除原设置
        allocOpt.flags_ |= ALLOC_FORCE_BY_RANK;
    } else {
        allocOpt.preferredRank_.push_back(RankId(options.policy));
    }
    return MMC_OK;
}

Result MmcClientDefault::Put(const std::string &key, const MmcBufferArray &bufArr, mmc_put_options &options,
                             uint32_t flags)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);
    MMC_ASSERT_LOG_AND_RETURN(!bufArr.Buffers().empty(), "bufArr.Buffers() is empty", MMC_ERROR);
    uint64_t operateId = GenerateOperateId(rankId_);
    AllocRequest request{key, {}, operateId};
    MMC_VALIDATE_RETURN(PrepareAllocOpt(bufArr.TotalSize(), options, flags, request.options_) == MMC_OK, "put error",
                        MMC_ERROR);
    AllocResponse response;
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " alloc " << key << " failed");
    if (response.result_ == MMC_DUPLICATED_OBJECT) {
        return response.result_;
    }
    MMC_RETURN_ERROR(response.result_, "client " << name_ << " alloc " << key << " failed");
    if (response.numBlobs_ == 0 || response.numBlobs_ != response.blobs_.size()) {
        MMC_LOG_ERROR("client " << name_ << " alloc " << key << " failed, blobs size:" << response.blobs_.size()
                                << ", numBlob:" << response.numBlobs_);
        return MMC_ERROR;
    }

    Result result = MMC_OK;
    BatchUpdateRequest updateRequest{};
    for (uint8_t i = 0; i < response.numBlobs_; i++) {
        auto blob = response.blobs_[i];
        MMC_LOG_DEBUG("Attempting to put to blob " << static_cast<int>(i) << " key " << key);
        TP_TRACE_BEGIN(TP_MMC_LOCAL_PUT);
        auto ret = bmProxy_->BatchPut(bufArr, blob);
        TP_TRACE_END(TP_MMC_LOCAL_PUT, 0);
        if (ret != MMC_OK) {
            MMC_LOG_ERROR("client " << name_ << " put " << key << " blob rank: " << blob.rank_
                                    << ", media: " << blob.mediaType_ << " failed, ret: " << ret);
            updateRequest.actionResults_.push_back(MMC_WRITE_FAIL);
            result = ret;
        } else {
            updateRequest.actionResults_.push_back(MMC_WRITE_OK);
        }

        updateRequest.keys_.push_back(key);
        updateRequest.ranks_.push_back(blob.rank_);
        updateRequest.mediaTypes_.push_back(blob.mediaType_);
        updateRequest.operateIds_.push_back(operateId);
    }
    Result updateRet = SyncUpdateState(updateRequest, "put update");
    if (updateRet != MMC_OK) {
        MMC_LOG_ERROR("client " << name_ << " put update state failed: " << updateRet);
        return updateRet;
    }
    return result;
}

Result MmcClientDefault::BatchPut(const std::vector<std::string> &keys, const std::vector<mmc_buffer> &bufs,
                                  mmc_put_options &options, uint32_t flags, std::vector<int> &batchResult)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (keys.empty() || bufs.empty() || keys.size() != bufs.size()) {
        MMC_LOG_ERROR("client " << name_ << " batch get failed: keys size:" << keys.size()
                                << ", bufs size:" << bufs.size());
        return MMC_INVALID_PARAM;
    }

    std::vector<MmcBufferArray> bufferArrays{};
    bufferArrays.reserve(bufs.size());
    for (const auto &buf : bufs) {
        MmcBufferArray bufArr{};
        bufArr.AddBuffer(buf);
        bufferArrays.emplace_back(bufArr);
    }

    return BatchPut(keys, bufferArrays, options, flags, batchResult);
}

Result MmcClientDefault::BatchPut(const std::vector<std::string> &keys, const std::vector<MmcBufferArray> &bufArrs,
                                  mmc_put_options &options, uint32_t flags, std::vector<int> &batchResult)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (keys.empty() || bufArrs.empty() || keys.size() != bufArrs.size()) {
        MMC_LOG_ERROR("client " << name_ << " batch get failed: keys size:" << keys.size()
                                << ", bufArrs size:" << bufArrs.size());
        return MMC_INVALID_PARAM;
    }

    // alloc blobs
    uint64_t operateId = GenerateOperateId(rankId_);
    BatchAllocRequest request(keys, {}, flags, operateId);
    for (const auto &bufArr : bufArrs) {
        AllocOptions tmpAllocOptions{};
        MMC_VALIDATE_RETURN(PrepareAllocOpt(bufArr.TotalSize(), options, flags, tmpAllocOptions) == MMC_OK,
                            "option param error", MMC_ERROR);
        request.options_.emplace_back(std::move(tmpAllocOptions));
    }
    BatchAllocResponse allocResponse{};
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, allocResponse, rpcRetryTimeOut_), "batch put alloc failed");
    // check alloc result
    if (keys.size() != allocResponse.blobs_.size() || keys.size() != allocResponse.numBlobs_.size() ||
        keys.size() != allocResponse.results_.size()) {
        MMC_LOG_ERROR("Mismatch in number of keys and allocated blobs, keys="
                      << keys.size() << ", blobs=" << allocResponse.blobs_.size() << ", numBlobs="
                      << allocResponse.numBlobs_.size() << ", results=" << allocResponse.results_.size());
        return MMC_ERROR;
    }

    // put obj — 从此时开始计时, alloc 失败不会产生虚假指标
    batchResult.assign(keys.size(), MMC_OK);
    BandwidthGuard guard(MetricOp::BATCH_PUT, bandwidthCollector_, bufArrs, batchResult);
    auto ret = PutData2Blobs(keys, bufArrs, allocResponse, batchResult);

    // update blob state
    BatchUpdateRequest updateRequest{};
    for (size_t i = 0; i < keys.size(); ++i) {
        for (const auto &blob : allocResponse.blobs_[i]) {
            updateRequest.keys_.push_back(keys[i]);
            updateRequest.ranks_.push_back(blob.rank_);
            updateRequest.mediaTypes_.push_back(blob.mediaType_);
            updateRequest.actionResults_.push_back(batchResult[i] == 0 ? MMC_WRITE_OK : MMC_WRITE_FAIL);
            updateRequest.operateIds_.push_back(operateId);
        }
    }
    // 写需要同步更新，异步更新会出现立即读查询blob不可读的情况
    Result updateRet = SyncUpdateState(updateRequest, "batch put update");
    if (updateRet != MMC_OK) {
        // WRITE_OK 未到服务端时 blob 停在 ALLOCATED，若仍返回成功会导致假写成功
        // 且 key 在租约超时前无法重写。报错让调用方感知，可通过 Remove(key) 后重写恢复。
        MMC_LOG_ERROR("client " << name_ << " batch put update state failed: " << updateRet);
        return updateRet;
    }

    if (ret != MMC_OK) {
        MMC_LOG_ERROR("client " << name_ << " batch put failed: " << ret);
    }
    return ret;
}

Result MmcClientDefault::Get(const char *key, mmc_buffer *buf, uint32_t flags)
{
    if (buf == nullptr || key == nullptr || key[0] == '\0' || strnlen(key, KEY_MAX_LENTH + 1) == KEY_MAX_LENTH + 1) {
        MMC_LOG_ERROR("Invalid arguments, buf=" << buf << ", key=" << static_cast<const void *>(key));
        return MMC_ERROR;
    }

    MmcBufferArray bufArr{};
    bufArr.AddBuffer(*buf);
    return Get(key, bufArr, flags);
}

Result MmcClientDefault::Get(const std::string &key, const MmcBufferArray &bufArr, uint32_t flags)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    uint64_t operateId = GenerateOperateId(rankId_);
    GetRequest request{key, rankId_, operateId, true};
    AllocResponse response;
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " get " << key << " failed");
    if (response.numBlobs_ == 0 || response.blobs_.empty()) {
        MMC_LOG_ERROR("client " << name_ << " get " << key
                                << " failed, numblob is:" << static_cast<uint64_t>(response.numBlobs_));
        return MMC_ERROR;
    }
    auto &blob = response.blobs_[0];
    uint64_t localLeaseDeadlineMs = ToLocalLeaseDeadlineMs(blob);
    TP_TRACE_BEGIN(TP_MMC_LOCAL_GET);
    auto ret = bmProxy_->BatchGet(bufArr, blob);
    TP_TRACE_END(TP_MMC_LOCAL_GET, 0);
    Result leaseCheckRet = CheckLeaseDeadline(localLeaseDeadlineMs);

    BatchUpdateRequest updateRequest{};
    updateRequest.actionResults_.push_back(MMC_READ_FINISH);
    updateRequest.keys_.push_back(key);
    updateRequest.ranks_.push_back(blob.rank_);
    updateRequest.mediaTypes_.push_back(blob.mediaType_);
    updateRequest.operateIds_.push_back(operateId);
    AsyncUpdateState(updateRequest, "get update");

    if (ret != MMC_OK) {
        MMC_LOG_ERROR("client " << name_ << " get " << key << " read data failed.");
        return ret;
    }
    if (leaseCheckRet != MMC_OK) {
        MMC_LOG_ERROR("client " << name_ << " get " << key << " lease expired after read.");
        return leaseCheckRet;
    }
    return MMC_OK;
}

Result MmcClientDefault::BatchGet(const std::vector<std::string> &keys, std::vector<mmc_buffer> &bufs, uint32_t flags,
                                  std::vector<int> &batchResult)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);
    if ((keys.empty() || bufs.empty() || keys.size() != bufs.size())) {
        MMC_LOG_ERROR("client " << name_ << " batch get failed: keys size:" << keys.size()
                                << ", bufArrs size:" << bufs.size());
        return MMC_INVALID_PARAM;
    }

    std::vector<MmcBufferArray> bufferArrays{};
    bufferArrays.reserve(bufs.size());
    for (const auto &buf : bufs) {
        MmcBufferArray bufArr{};
        bufArr.AddBuffer(buf);
        bufferArrays.emplace_back(bufArr);
    }
    return BatchGet(keys, bufferArrays, flags, batchResult);
}

Result MmcClientDefault::BatchGet(const std::vector<std::string> &keys, const std::vector<MmcBufferArray> &bufArrs,
                                  uint32_t flags, std::vector<int> &batchResult)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if ((keys.empty() || bufArrs.empty() || keys.size() != bufArrs.size())) {
        MMC_LOG_ERROR("client " << name_ << " batch get failed: keys size:" << keys.size()
                                << ", bufArrs size:" << bufArrs.size());
        return MMC_INVALID_PARAM;
    }
    // get meta
    batchResult.assign(keys.size(), MMC_INVALID_PARAM);
    const uint64_t operateId = GenerateOperateId(rankId_);
    BatchGetRequest request{keys, rankId_, operateId};
    BatchAllocResponse response;
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " batch get failed");
    // check rsp meta
    if (response.blobs_.size() != keys.size() || response.numBlobs_.size() != keys.size()) {
        MMC_LOG_ERROR("client " << name_ << " batch get response size mismatch: expected " << keys.size()
                                << ", got blobs=" << response.blobs_.size()
                                << ", numBlobs=" << response.numBlobs_.size());
        return MMC_ERROR;
    }
    // read data — 从此时开始计时, alloc 失败不会产生虚假指标
    BandwidthGuard guard(MetricOp::BATCH_GET, bandwidthCollector_, bufArrs, batchResult);
    std::vector<uint64_t> localLeaseDeadlinesMs = ToLocalLeaseDeadlinesMs(response.blobs_);
    MediaType mediaType = MEDIA_NONE;
    std::vector<std::tuple<uint32_t, uint32_t, std::future<int32_t>>> futures;
    size_t startKeyIndex = 0;
    BatchCopyDesc copyDesc{};
    for (size_t i = 0; i < keys.size(); ++i) {
        const MmcBufferArray &bufArr = bufArrs[i];
        const auto &blobs = response.blobs_[i];
        uint8_t numBlobs = response.numBlobs_[i];
        if (numBlobs <= 0 || blobs.empty() || blobs.size() != numBlobs) {
            MMC_LOG_ERROR("client " << name_ << " batch get failed for key " << keys[i]
                                    << ", blob:" << std::to_string(numBlobs) << ", size:" << blobs.size());
            batchResult[i] = MMC_OBJECT_NOT_EXISTS;
            continue;
        }
        if (bufArr.TotalSize() != blobs[0].size_) {
            MMC_LOG_ERROR("client " << name_ << " batch get failed for key " << keys[i]
                                    << ", blob:" << std::to_string(numBlobs) << ", size:" << blobs[0].size_
                                    << " key size:" << bufArr.TotalSize());
            batchResult[i] = MMC_INTERNAL_ERROR;
            continue;
        }

        BatchCopyDesc keyCopyDesc{};
        batchResult[i] = PrepareBlob(bufArr, blobs[0], mediaType, keyCopyDesc, true);
        if (batchResult[i] != MMC_OK) {
            MMC_LOG_ERROR("client " << name_ << " prepare blob failed for key " << keys[i]
                                    << ", ret=" << batchResult[i]);
            continue;
        }
        copyDesc.Append(keyCopyDesc);
        /**
         * 1. 开启聚合是避免单流读取地址太少，避免调用栈开销
         * 2. 也要避免聚合io聚合地址太多，不利于并发，实测单流性能低于多流
         */
        if (aggregateIO_ && (copyDesc.sizes.size() < aggregateNum_) && (i != (keys.size() - 1))) {
            continue;
        }
        auto future = SubmitGetTask(copyDesc, mediaType, !(startKeyIndex == 0 && i == (keys.size() - 1)));
        futures.push_back(std::make_tuple(startKeyIndex, i, std::move(future)));
        copyDesc.Clear();
        startKeyIndex = i + 1;
    }
    //  last key is invalid, task not submit
    if (!copyDesc.sizes.empty()) {
        auto future = SubmitGetTask(copyDesc, mediaType, !(startKeyIndex == 0));
        futures.push_back(std::make_tuple(startKeyIndex, (keys.size() - 1), std::move(future)));
    }

    TP_TRACE_BEGIN(TP_MMC_LOCAL_GET_WAIT_FUTURE);
    WaitFeatures(futures, batchResult);
    TP_TRACE_END(TP_MMC_LOCAL_GET_WAIT_FUTURE, 0);
    bool hasExpiredLease = false;
    const uint64_t leaseCheckNowMs = NowMs();
    for (size_t i = 0; i < localLeaseDeadlinesMs.size() && i < batchResult.size(); ++i) {
        if (localLeaseDeadlinesMs[i] != 0 && leaseCheckNowMs > localLeaseDeadlinesMs[i] && batchResult[i] == MMC_OK) {
            batchResult[i] = MMC_LEASE_EXPIRED;
            hasExpiredLease = true;
        }
    }
    // update read state
    BatchUpdateRequest updateRequest{};
    for (size_t i = 0; i < keys.size(); ++i) {
        for (const auto &blob : response.blobs_[i]) {
            updateRequest.keys_.push_back(keys[i]);
            updateRequest.ranks_.push_back(blob.rank_);
            updateRequest.mediaTypes_.push_back(blob.mediaType_);
            updateRequest.actionResults_.push_back(MMC_READ_FINISH);
            updateRequest.operateIds_.push_back(operateId);
        }
    }
    AsyncUpdateState(updateRequest, "batch get update");
    if (hasExpiredLease) {
        MMC_LOG_ERROR("client " << name_ << " batch get lease expired.");
        return MMC_LEASE_EXPIRED;
    }
    return MMC_OK;
}

Result MmcClientDefault::Remove(const char *key, uint32_t flags)
{
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    RemoveRequest request{key};
    Response response;
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " remove " << key << " failed");
    if (response.ret_ == MMC_OK) {
        gvaBlobTracker_.RemoveByKey(std::string(key));
    }
    return response.ret_;
}

Result MmcClientDefault::BatchRemove(const std::vector<std::string> &keys, std::vector<Result> &remove_results,
                                     uint32_t flags)
{
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    BatchRemoveRequest request{keys};
    BatchRemoveResponse response;

    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " BatchRemove failed");

    if (response.results_.size() != keys.size()) {
        MMC_LOG_ERROR("BatchRemove response size mismatch. Expected: " << keys.size()
                                                                       << ", Got: " << response.results_.size());
        std::fill(remove_results.begin(), remove_results.end(), MMC_ERROR);
        return MMC_ERROR;
    }

    remove_results = response.results_;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (remove_results[i] == MMC_OK) {
            gvaBlobTracker_.RemoveByKey(keys[i]);
        }
    }
    return MMC_OK;
}

Result MmcClientDefault::RemoveAll(uint32_t flags)
{
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    RemoveAllRequest request{};
    Response response;

    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " RemoveAll failed");

    gvaBlobTracker_.Clear();
    return MMC_OK;
}

Result MmcClientDefault::IsExist(const std::string &key, uint32_t flags) const
{
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (key.empty()) {
        MMC_LOG_ERROR("Get empty key!");
        return MMC_INVALID_PARAM;
    }

    IsExistRequest request{key};
    Response response;
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " IsExist " << key << " failed");
    return response.ret_;
}

Result MmcClientDefault::BatchIsExist(const std::vector<std::string> &keys, std::vector<int32_t> &exist_results,
                                      uint32_t flags) const
{
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (keys.empty()) {
        MMC_LOG_ERROR("Get empty keys!");
        return MMC_INVALID_PARAM;
    }

    BatchIsExistRequest request{keys};
    BatchIsExistResponse response;
    BandwidthGuard guard(MetricOp::BATCH_EXIST, bandwidthCollector_);
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " BatchIsExist failed");

    if (response.results_.size() != keys.size()) {
        MMC_LOG_ERROR("BatchIsExist response size mismatch. Expected: " << keys.size()
                                                                        << ", Got: " << response.results_.size());
        std::fill(exist_results.begin(), exist_results.end(), MMC_ERROR);
        return MMC_ERROR;
    }

    exist_results = response.results_;
    return MMC_OK;
}

Result MmcClientDefault::Query(const std::string &key, mmc_data_info &query_info, uint32_t flags)
{
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (key.empty()) {
        MMC_LOG_ERROR("Get empty key!");
        return MMC_INVALID_PARAM;
    }

    const uint64_t operateId = GenerateOperateId(rankId_);
    QueryRequest request{key, operateId, 0};
    request.flag_ |= flags;
    QueryResponse response;
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " Query " << key << " failed");
    query_info.size = response.queryInfo_.size_;
    query_info.prot = response.queryInfo_.prot_;
    const size_t queryBlobCount =
        std::min(std::min(static_cast<size_t>(response.queryInfo_.numBlobs_), response.queryInfo_.blobs_.size()),
                 static_cast<size_t>(MAX_BLOB_COPIES));
    query_info.numBlobs = static_cast<uint8_t>(queryBlobCount);
    query_info.valid = response.queryInfo_.valid_;
    for (size_t i = 0; i < queryBlobCount; i++) {
        query_info.ranks[i] = response.queryInfo_.blobs_[i].rank_;
        query_info.types[i] = response.queryInfo_.blobs_[i].mediaType_;
        query_info.gvas[i] = response.queryInfo_.blobs_[i].gva_;
    }
    return MMC_OK;
}

Result MmcClientDefault::BatchQuery(const std::vector<std::string> &keys, std::vector<mmc_data_info> &query_infos,
                                    uint32_t flags)
{
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (keys.empty()) {
        MMC_LOG_ERROR("Get empty keys!");
        return MMC_INVALID_PARAM;
    }

    const uint64_t operateId = GenerateOperateId(rankId_);
    BatchQueryRequest request{keys, operateId, 0};
    request.flag_ |= flags;
    BatchQueryResponse response;
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " BatchIsExist failed");

    if (response.batchQueryInfos_.size() != keys.size()) {
        MMC_LOG_ERROR("BatchQuery get a response with mismatched size ("
                      << response.batchQueryInfos_.size() << "), should get size (" << keys.size() << ").");
        query_infos.resize(keys.size(), {});
        return MMC_ERROR;
    }

    for (size_t idx = 0; idx < response.batchQueryInfos_.size(); ++idx) {
        const auto &info = response.batchQueryInfos_[idx];
        mmc_data_info outInfo{};
        outInfo.valid = info.valid_;
        if (!outInfo.valid) {
            query_infos.push_back(outInfo);
            continue;
        }

        const size_t queryBlobCount = std::min(std::min(static_cast<size_t>(info.numBlobs_), info.blobs_.size()),
                                               static_cast<size_t>(MAX_BLOB_COPIES));
        for (size_t i = 0; i < queryBlobCount; i++) {
            outInfo.ranks[i] = info.blobs_[i].rank_;
            outInfo.types[i] = info.blobs_[i].mediaType_;
            outInfo.gvas[i] = info.blobs_[i].gva_;
        }
        outInfo.size = info.size_;
        outInfo.prot = info.prot_;
        outInfo.numBlobs = static_cast<uint8_t>(queryBlobCount);
        query_infos.push_back(outInfo);
    }
    return MMC_OK;
}

Result MmcClientDefault::PrefetchKeys(std::vector<std::string> keys, MediaType srcMedia, MediaType dstMedia,
                                      uint32_t flags)
{
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (keys.empty()) {
        MMC_LOG_ERROR("client " << name_ << " prefetch invalid input, keys size: " << keys.size());
        return MMC_INVALID_PARAM;
    }

    PrefetchRequest request{keys, static_cast<uint16_t>(srcMedia), static_cast<uint16_t>(dstMedia), flags};
    PrefetchResponse response;
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " PrefetchKeys failed");
    return response.ret_;
}

Result MmcClientDefault::EvictKeys(std::vector<std::string> keys, MediaType srcMedia, MediaType dstMedia,
                                   uint32_t flags)
{
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (keys.empty()) {
        MMC_LOG_ERROR("client " << name_ << " evict invalid input, keys size: " << keys.size());
        return MMC_INVALID_PARAM;
    }

    EvictRequest request{keys, static_cast<uint16_t>(srcMedia), static_cast<uint16_t>(dstMedia), flags};
    EvictResponse response;
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_),
                     "client " << name_ << " EvictKeys failed");
    return response.ret_;
}

Result MmcClientDefault::BatchAddLease(const std::vector<std::string> &keys, uint64_t leaseTtlMs,
                                       std::vector<int> &results)
{
    return BatchAddLeaseImpl(keys, leaseTtlMs, results, nullptr);
}

Result MmcClientDefault::BatchAddLeaseWithInfo(const std::vector<std::string> &keys, uint64_t leaseTtlMs,
                                               std::vector<int> &results, std::vector<mmc_data_info> &infos)
{
    return BatchAddLeaseImpl(keys, leaseTtlMs, results, &infos);
}

Result MmcClientDefault::BatchAddLeaseImpl(const std::vector<std::string> &keys, uint64_t leaseTtlMs,
                                           std::vector<int> &results, std::vector<mmc_data_info> *infos)
{
    results.assign(keys.size(), MMC_INVALID_PARAM);
    if (infos != nullptr) {
        infos->assign(keys.size(), {});
    }
    if (metaNetClient_ == nullptr) {
        results.assign(keys.size(), MMC_CLIENT_NOT_INIT);
        MMC_LOG_ERROR("MetaNetClient is null");
        return MMC_CLIENT_NOT_INIT;
    }
    if (keys.empty()) {
        MMC_LOG_ERROR("client " << name_ << " batch add lease invalid input, key size:" << keys.size());
        return MMC_INVALID_PARAM;
    }

    const uint64_t newOperateId = GenerateOperateId(rankId_);
    std::vector<uint64_t> operateIds;
    operateIds.resize(keys.size(), newOperateId);
    BatchUpdateLeaseRequest request{keys, operateIds, leaseTtlMs};
    BatchUpdateLeaseResponse response;
    Result rpcRet = metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_);
    if (rpcRet != MMC_OK) {
        MMC_LOG_ERROR("client " << name_ << " batch add lease failed");
        results.assign(keys.size(), rpcRet);
        return rpcRet;
    }
    if (response.ret_ != MMC_OK) {
        results.assign(keys.size(), response.ret_);
        return response.ret_;
    }

    if (response.results_.size() != keys.size() || response.batchQueryInfos_.size() != keys.size()) {
        MMC_LOG_ERROR("client " << name_ << " batch add lease response size mismatch: expected " << keys.size()
                                << ", result size:" << response.results_.size()
                                << ", info size:" << response.batchQueryInfos_.size());
        results.assign(keys.size(), MMC_ERROR);
        return MMC_ERROR;
    }

    results = response.results_;
    for (size_t i = 0; i < response.batchQueryInfos_.size(); ++i) {
        if (results[i] != MMC_OK) {
            continue;
        }
        const auto &queryInfo = response.batchQueryInfos_[i];
        if (!queryInfo.valid_ || queryInfo.numBlobs_ != 1U || queryInfo.blobs_.size() != 1U) {
            MMC_LOG_ERROR("client " << name_ << " batch add lease got invalid query info for key " << keys[i]
                                    << ", valid=" << queryInfo.valid_
                                    << ", numBlobs=" << static_cast<uint32_t>(queryInfo.numBlobs_)
                                    << ", blobsSize=" << queryInfo.blobs_.size());
            results[i] = MMC_ERROR;
            continue;
        }

        const auto &blob = queryInfo.blobs_[0];
        Result trackRet = gvaBlobTracker_.UpdateFromQuery(keys[i], blob, operateIds[i], ToLocalLeaseDeadlineMs(blob));
        MMC_LOG_DEBUG("UpdateFromQuery key:" << keys[i] << ", leaseTime:" << blob.leaseTimeoutTtlMs_);
        if (trackRet != MMC_OK) {
            MMC_LOG_ERROR("client " << name_ << " batch add lease track failed for key " << keys[i]
                                    << ", ret:" << trackRet);
            results[i] = trackRet;
        }
        TP_TRACE_RECORD(TP_MMC_TRACKER_ADD_LEASE_FROM_READ, 1000ULL, trackRet);
        if (infos != nullptr && results[i] == MMC_OK) {
            auto &info = (*infos)[i];
            info.valid = true;
            info.size = queryInfo.size_;
            info.prot = queryInfo.prot_;
            info.numBlobs = queryInfo.numBlobs_;
            info.ranks[0] = blob.rank_;
            info.types[0] = blob.mediaType_;
            info.gvas[0] = blob.gva_;
        }
    }
    return MMC_OK;
}

Result MmcClientDefault::BatchRemoveLease(const std::vector<std::string> &keys)
{
    if (metaNetClient_ == nullptr) {
        MMC_LOG_ERROR("MetaNetClient is null");
        return MMC_CLIENT_NOT_INIT;
    }
    if (keys.empty()) {
        MMC_LOG_ERROR("client " << name_ << " batch remove lease invalid input, key size:" << keys.size());
        return MMC_INVALID_PARAM;
    }
    std::vector<uint64_t> operateIds;
    operateIds.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        LocalGvaBlobInfo info{};
        Result findRet = gvaBlobTracker_.FindReadLeaseByKey(keys[i], info);
        if (findRet != MMC_OK) {
            MMC_LOG_WARN("client " << name_ << " batch remove lease find local gva info failed, key:" << keys[i]
                                   << " because of lease timeout, ret:" << findRet);
            return MMC_LEASE_EXPIRED;
        }
        auto innerOpId = gvaBlobTracker_.ReleaseLease(keys[i]);
        if (innerOpId == UINT64_MAX) {
            MMC_LOG_DEBUG("key " << keys[i] << " inner operateId_:" << innerOpId << " remove keys != alloc keys ");
        }
        TP_TRACE_RECORD(TP_MMC_TRACKER_REMOVE_LEASE_FROM_READ, 1000ULL, innerOpId == UINT64_MAX ? -1 : 0);
        operateIds.push_back(innerOpId);
    }

    BatchUpdateLeaseRequest request{keys, operateIds, 0, 1};
    AsyncUpdateLease(request);
    return MMC_OK;
}

void MmcClientDefault::WaitFeatures(std::vector<std::tuple<uint32_t, uint32_t, std::future<int32_t>>> &futures,
                                    std::vector<int> &batchResult)
{
    for (auto &tuple : futures) {
        auto res = std::get<2>(tuple).get();
        if (res == MMC_OK) {
            continue;
        }
        auto start = std::get<0>(tuple);
        auto end = std::get<1>(tuple);
        MMC_LOG_ERROR("batch key from " << start << " to " << end << " failed, error code " << res);
        for (size_t i = start; i <= end && i < batchResult.size(); i++) {
            if (batchResult[i] == MMC_OK) {
                batchResult[i] = res;
            }
        }
    }
}

void MmcClientDefault::ProcessUbsIoBatchGetWithHBM(UbsIoBatchGetData &data)
{
    std::vector<size_t> ubsIoIndices;
    data.ubsIoKeys.reserve(data.keys.size());
    ubsIoIndices.reserve(data.keys.size());
    std::vector<std::vector<void *>> npuBufAddrs;
    std::vector<std::vector<size_t>> npuBufLengths;
    npuBufAddrs.reserve(data.keys.size());
    npuBufLengths.reserve(data.keys.size());
    for (size_t i = 0; i < data.keys.size(); ++i) {
        if (data.batchResult[i] == MMC_ERROR) {
            data.ubsIoKeys.emplace_back(data.keys[i]);
            ubsIoIndices.emplace_back(i);
            auto &keyBuffers = data.bufArrs[i].Buffers();
            std::vector<void *> npuBufAddrsForThisKey;
            std::vector<size_t> npuBufLengthsForThisKey;
            npuBufAddrsForThisKey.reserve(keyBuffers.size());
            npuBufLengthsForThisKey.reserve(keyBuffers.size());
            for (auto &buffer : keyBuffers) {
                npuBufAddrsForThisKey.emplace_back(reinterpret_cast<void *>(buffer.addr + buffer.offset));
                npuBufLengthsForThisKey.emplace_back(buffer.len);
            }
            npuBufAddrs.emplace_back(std::move(npuBufAddrsForThisKey));
            npuBufLengths.emplace_back(std::move(npuBufLengthsForThisKey));
        }
    }
    if (!data.ubsIoKeys.empty()) {
        TP_TRACE_BEGIN(TP_MMC_CLIENT_UBS_IO_BATCH_GET);
        std::vector<int> ubsIoResults(data.ubsIoKeys.size(), 0);

        TP_TRACE_BEGIN(TP_MMC_CLIENT_UBS_IO_BATCH_GET_2);
        Result ubsIoRet = ubsIoProxy_->BatchGetWithHBM(data.ubsIoKeys, npuBufAddrs, npuBufLengths, ubsIoResults);
        TP_TRACE_END(TP_MMC_CLIENT_UBS_IO_BATCH_GET_2, MMC_OK);
        if (ubsIoRet != MMC_OK) {
            MMC_LOG_ERROR("ubsIo batch get failed, ret: " << ubsIoRet);
            return;
        }
        for (size_t i = 0; i < data.ubsIoKeys.size(); ++i) {
            size_t originIndex = ubsIoIndices[i];
            if (ubsIoResults[i] != 0) {
                MMC_LOG_ERROR("ubsIo batch get failed for key " << data.ubsIoKeys[i]
                                                                << ", result: " << ubsIoResults[i]);
                data.batchResult[originIndex] = MMC_ERROR;
            } else {
                data.batchResult[originIndex] = MMC_OK;
            }
        }
        TP_TRACE_END(TP_MMC_CLIENT_UBS_IO_BATCH_GET, MMC_OK);
    }
}

Result MmcClientDefault::SyncUpdateState(BatchUpdateRequest &updateRequest, const std::string &opName)
{
    TP_TRACE_BEGIN(TP_MMC_LOCAL_BATCH_UPDATE);
    BatchUpdateResponse updateResponse;
    Result updateResult = metaNetClient_->SyncCall(updateRequest, updateResponse, rpcRetryTimeOut_);
    TP_TRACE_END(TP_MMC_LOCAL_BATCH_UPDATE, updateResult);
    if (updateResult != MMC_OK || updateResponse.results_.size() != updateRequest.keys_.size()) {
        MMC_LOG_ERROR("client " << name_ << " " << opName << " failed:" << updateResult << ", key size:"
                                << updateRequest.keys_.size() << ", ret size:" << updateResponse.results_.size());
        return updateResult != MMC_OK ? updateResult : MMC_ERROR;
    }
    for (size_t i = 0; i < updateRequest.keys_.size(); ++i) {
        if (updateResponse.results_[i] != MMC_OK) {
            MMC_LOG_ERROR("client " << name_ << " " << opName << " for key " << updateRequest.keys_[i]
                                    << " failed:" << updateResponse.results_[i]);
        }
    }
    return MMC_OK;
}

void MmcClientDefault::AsyncUpdateState(BatchUpdateRequest &updateRequest, const std::string &opName)
{
    auto future = threadPool_->Enqueue(
        [this, opName](BatchUpdateRequest updateRequestL) { SyncUpdateState(updateRequestL, opName); }, updateRequest);
    if (!future.valid()) {
        SyncUpdateState(updateRequest, opName);
    }
}

void MmcClientDefault::SyncUpdateLease(BatchUpdateLeaseRequest &request)
{
    TP_TRACE_BEGIN(TP_MMC_LOCAL_BATCH_UPDATE);
    BatchUpdateLeaseResponse response;
    Result updateResult = metaNetClient_->SyncCall(request, response, rpcRetryTimeOut_);
    TP_TRACE_END(TP_MMC_LOCAL_BATCH_UPDATE, updateResult);
    if (updateResult != MMC_OK) {
        MMC_LOG_ERROR("client " << name_ << " batch update lease failed:" << updateResult);
        return;
    }
    if (response.ret_ != MMC_OK) {
        MMC_LOG_ERROR("client " << name_ << " batch update lease response failed:" << response.ret_);
        return;
    }
    if (response.results_.size() != request.keys_.size()) {
        MMC_LOG_ERROR("client " << name_ << " batch update lease response size mismatch, key size:"
                                << request.keys_.size() << ", ret size:" << response.results_.size());
        return;
    }
    for (size_t i = 0; i < request.keys_.size() && i < response.results_.size(); ++i) {
        if (response.results_[i] != MMC_OK) {
            MMC_LOG_ERROR("client " << name_ << " batch update lease for key " << request.keys_[i]
                                    << " failed:" << response.results_[i]);
        }
    }
}

void MmcClientDefault::AsyncUpdateLease(BatchUpdateLeaseRequest &request)
{
    auto future = threadPool_->Enqueue([&](BatchUpdateLeaseRequest requestL) { SyncUpdateLease(requestL); }, request);
    if (!future.valid()) {
        SyncUpdateLease(request);
    }
}

Result MmcClientDefault::InitMetricReporting()
{
    if (!bmProxy_->IsReady()) {
        return MMC_OK;
    }
    auto &metricMgr = MmcClientMetricManager::GetInstance();
    bandwidthCollector_ = metricMgr.InitDefaultCollectors();

    MMC_RETURN_ERROR(RegisterPeriodicTask("client_metric_report", CLIENT_METRIC_REPORT_INTERVAL_SECONDS,
                                          [this]() { ReportMetrics(); }),
                     "Failed to register client metric report task");
    return MMC_OK;
}

void MmcClientDefault::ReportMetrics()
{
    auto &metricMgr = MmcClientMetricManager::GetInstance();
    auto snapshot = metricMgr.CollectAll();
    StatsReportRequest req;
    req.rank_ = rankId_;
    for (size_t i = 0; i < static_cast<size_t>(MetricOp::COUNT); ++i) {
        req.bandwidths_[i] = snapshot.bandwidths[i];
    }
    req.ubsIo_ = snapshot.ubsIo;
    StatsReportResponse resp;
    const Result ret = metaNetClient_->SyncCall(req, resp, rpcRetryTimeOut_);
    if (ret != MMC_OK) {
        MMC_LOG_WARN("Failed to report client metrics, ret: " << ret
                                                              << ", window data will be accumulated into next report");
    } else {
        metricMgr.ResetAll();
    }
}

void MmcClientDefault::ProcessExpiredReadLeases()
{
    gvaBlobTracker_.RemoveExpired();
}

Result MmcClientDefault::PrepareBlob(const MmcBufferArray &bufArr, const MmcMemBlobDesc &blob, MediaType &mediaType,
                                     BatchCopyDesc &copyDesc, bool blobIsSrc)
{
    if (bufArr.Buffers().empty()) {
        MMC_LOG_ERROR("buffer is empty");
        return MMC_INVALID_PARAM;
    }
    uint64_t shift = 0;
    for (size_t k = 0; k < bufArr.Buffers().size(); ++k) {
        auto buf = &bufArr.Buffers()[k];
        if (buf->type == MEDIA_NONE) {
            MMC_LOG_ERROR("unexcepted buf type:" << buf->type);
            return MMC_INVALID_PARAM;
        }
        if (mediaType == MEDIA_NONE) {
            mediaType = static_cast<MediaType>(buf->type);
        } else if (mediaType != buf->type) {
            MMC_LOG_ERROR("not all data type same as " << mediaType << ", unexcepted buf type:" << buf->type);
            return MMC_INVALID_PARAM;
        }
        if (blobIsSrc) {
            copyDesc.dsts.push_back(reinterpret_cast<void *>(buf->addr + buf->offset));
            copyDesc.srcs.push_back(reinterpret_cast<void *>(blob.gva_ + shift));
        } else {
            copyDesc.srcs.push_back(reinterpret_cast<void *>(buf->addr + buf->offset));
            copyDesc.dsts.push_back(reinterpret_cast<void *>(blob.gva_ + shift));
        }
        copyDesc.sizes.push_back(buf->len);
        shift += MmcBufSize(*buf);
    }
    return MMC_OK;
}

Result MmcClientDefault::PrepareMultiBlobs(const MmcBufferArray &bufArr, const std::vector<MmcMemBlobDesc> &blobs,
                                           MediaType &mediaType, BatchCopyDesc &copyDesc, bool blobIsSrc)
{
    for (uint8_t j = 0; j < blobs.size(); ++j) {
        auto ret = PrepareBlob(bufArr, blobs[j], mediaType, copyDesc, blobIsSrc);
        if (ret != MMC_OK) {
            return ret;
        }
    }
    return MMC_OK;
}

std::future<int32_t> MmcClientDefault::SubmitPutTask(BatchCopyDesc &copyDesc, MediaType mediaType, bool asyncExec)
{
    if (asyncExec) {
        auto future = writeThreadPool_->Enqueue(
            [&](BatchCopyDesc copyDescL, MediaType localMediaL) -> int32_t {
                return bmProxy_->BatchDataPut(copyDescL.srcs, copyDescL.dsts, copyDescL.sizes, localMediaL);
            },
            copyDesc, mediaType);
        if (future.valid()) {
            return future;
        }
    }

    // 提交失败 或 直接执行
    std::promise<int32_t> prom{};
    std::future<int32_t> future = prom.get_future();
    TP_TRACE_BEGIN(TP_MMC_LOCAL_BATCH_PUT_SYNC);
    auto ret = bmProxy_->BatchDataPut(copyDesc.srcs, copyDesc.dsts, copyDesc.sizes, mediaType);
    TP_TRACE_END(TP_MMC_LOCAL_BATCH_PUT_SYNC, ret);
    prom.set_value(ret);
    return future;
}

std::future<int32_t> MmcClientDefault::SubmitGetTask(BatchCopyDesc &copyDesc, MediaType mediaType, bool asyncExec)
{
    if (asyncExec) {
        auto future = readThreadPool_->Enqueue(
            [&](BatchCopyDesc copyDescL, MediaType localMediaL) -> int32_t {
                return bmProxy_->BatchDataGet(copyDescL.srcs, copyDescL.dsts, copyDescL.sizes, localMediaL);
            },
            copyDesc, mediaType);
        if (future.valid()) {
            return future;
        }
    }
    // 提交失败 或 直接执行
    std::promise<int32_t> prom{};
    std::future<int32_t> future = prom.get_future();
    TP_TRACE_BEGIN(TP_MMC_LOCAL_BATCH_GET_SYNC);
    auto ret = bmProxy_->BatchDataGet(copyDesc.srcs, copyDesc.dsts, copyDesc.sizes, mediaType);
    TP_TRACE_END(TP_MMC_LOCAL_BATCH_GET_SYNC, ret);
    prom.set_value(ret);
    return future;
}

Result MmcClientDefault::PutData2Blobs(const std::vector<std::string> &keys, const std::vector<MmcBufferArray> &bufArrs,
                                       const BatchAllocResponse &allocResponse, std::vector<int> &batchResult)
{
    MediaType mediaType = MEDIA_NONE;
    std::vector<std::tuple<uint32_t, uint32_t, std::future<int32_t>>> futures;
    BatchCopyDesc copyDesc{};
    size_t startKeyIndex = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string &key = keys[i];
        const MmcBufferArray &bufArr = bufArrs[i];
        const auto &blobs = allocResponse.blobs_[i];
        const auto numBlobs = allocResponse.numBlobs_[i];
        if (allocResponse.results_[i] != MMC_OK) {
            // alloc has error, reserve alloc error code
            if (allocResponse.results_[i] != MMC_DUPLICATED_OBJECT) {
                MMC_LOG_ERROR("Alloc blob failed for key " << key << ", error code=" << allocResponse.results_[i]);
            }
            batchResult[i] = allocResponse.results_[i];
            continue;
        } else if (numBlobs == 0 || blobs.size() != numBlobs) {
            MMC_LOG_ERROR("Invalid number of blobs" << numBlobs << " , " << blobs.size() << " for key " << key);
            // batchResult 初始值为 MMC_OK，不设置错误码会向服务端发送 WRITE_OK，
            // 导致未写入数据的 blob 被置为 READABLE
            batchResult[i] = MMC_ERROR;
            continue;
        }

        // 一个key对应的所有blob副本
        BatchCopyDesc keyCopyDesc{};
        batchResult[i] = PrepareMultiBlobs(bufArr, blobs, mediaType, keyCopyDesc, false);
        if (batchResult[i] != MMC_OK) {
            MMC_LOG_ERROR("Prepare multi blobs failed for key " << key << ", ret=" << batchResult[i]);
            continue;
        }

        copyDesc.Append(keyCopyDesc);
        if (aggregateIO_ && (copyDesc.sizes.size() < aggregateNum_) && (i != (keys.size() - 1))) {
            continue;
        }

        auto future = SubmitPutTask(copyDesc, mediaType, !(startKeyIndex == 0 && i == (keys.size() - 1)));
        futures.push_back(std::make_tuple(startKeyIndex, i, std::move(future)));

        copyDesc.Clear();
        startKeyIndex = i + 1;
    }
    //  last key is invalid, task not submit
    if (!copyDesc.sizes.empty()) {
        auto future = SubmitPutTask(copyDesc, mediaType, !(startKeyIndex == 0));
        futures.push_back(std::make_tuple(startKeyIndex, (keys.size() - 1), std::move(future)));
    }

    TP_TRACE_BEGIN(TP_MMC_LOCAL_PUT_WAIT_FUTURE);
    WaitFeatures(futures, batchResult);
    TP_TRACE_END(TP_MMC_LOCAL_PUT_WAIT_FUTURE, 0);

    return MMC_OK;
}

Result MmcClientDefault::RegisterBuffer(uint64_t addr, uint64_t size)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    return bmProxy_->RegisterBuffer(addr, size);
}

Result MmcClientDefault::UnRegisterBuffer(uint64_t addr, uint64_t size)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    return bmProxy_->UnRegisterBuffer(addr);
}

Result MmcClientDefault::BatchMalloc(const std::vector<std::string> &keys, const std::vector<size_t> &sizes,
                                     const mmc_put_options &options, uint64_t leaseTtlMs, std::vector<uintptr_t> &gvas)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (keys.empty() || sizes.empty() || keys.size() != sizes.size()) {
        MMC_LOG_ERROR("client " << name_ << " batch get failed: keys size:" << keys.size()
                                << ", bufArrs size:" << sizes.size());
        return MMC_INVALID_PARAM;
    }

    // alloc blobs
    uint32_t flags = ALLOC_RANDOM | ALLOC_FLAGS_GVA_MALLOC_MASK;
    uint64_t operateId = GenerateOperateId(rankId_);
    BatchAllocRequest request(keys, {}, flags, operateId, leaseTtlMs);
    for (const auto &size : sizes) {
        AllocOptions tmpAllocOptions{};
        MMC_VALIDATE_RETURN(PrepareAllocOpt(size, options, flags, tmpAllocOptions) == MMC_OK, "option param error",
                            MMC_ERROR);
        request.options_.emplace_back(std::move(tmpAllocOptions));
    }
    BatchAllocResponse allocResponse{};
    MMC_RETURN_ERROR(metaNetClient_->SyncCall(request, allocResponse, rpcRetryTimeOut_), "batch put alloc failed");
    // check alloc result
    if (keys.size() != allocResponse.blobs_.size() || keys.size() != allocResponse.numBlobs_.size() ||
        keys.size() != allocResponse.results_.size()) {
        MMC_LOG_ERROR("Mismatch in number of keys and allocated blobs, keys="
                      << keys.size() << ", blobs=" << allocResponse.blobs_.size() << ", numBlobs="
                      << allocResponse.numBlobs_.size() << ", results=" << allocResponse.results_.size());
        return MMC_ERROR;
    }

    gvas.resize(keys.size(), 0);
    for (size_t i = 0; i < keys.size(); ++i) {
        const std::string &key = keys[i];
        const auto &blobs = allocResponse.blobs_[i];
        const auto numBlobs = allocResponse.numBlobs_[i];
        if (allocResponse.results_[i] != MMC_OK) {
            // alloc has error, reserve alloc error code
            if (allocResponse.results_[i] != MMC_DUPLICATED_OBJECT) {
                MMC_LOG_ERROR("Alloc blob failed for key " << key << ", error code=" << allocResponse.results_[i]);
                continue;
            }
        } else if (numBlobs == 0 || blobs.size() != numBlobs) {
            MMC_LOG_ERROR("Invalid number of blobs" << numBlobs << " , " << blobs.size() << " for key " << key);
            continue;
        }

        gvas[i] = blobs[0].gva_;
        Result trackRet = gvaBlobTracker_.RegisterFromBatchAlloc(key, blobs[0], operateId);
        MMC_LOG_DEBUG("BatchMalloc key:" << keys[i] << ", leaseTime:" << blobs[0].leaseTimeoutTtlMs_);
        if (trackRet != MMC_OK) {
            MMC_LOG_ERROR("Register batch alloc gva info failed for key " << key << ", ret:" << trackRet);
        }
        TP_TRACE_RECORD(TP_MMC_TRACKER_ADD_LEASE_FROM_WRITE, 1000ULL, trackRet);
    }
    return MMC_OK;
}

Result MmcClientDefault::BatchCopyWritePath(std::vector<void *> &gvas, std::vector<void *> &buffers,
                                            std::vector<size_t> &sizes, int32_t direct)
{
    std::vector<void *> toGvas{};
    std::vector<void *> toBuffers{};
    std::vector<size_t> toSizes{};

    for (size_t i = 0; i < gvas.size(); ++i) {
        LocalGvaBlobInfo info{};
        Result findRet = gvaBlobTracker_.FindWritable(reinterpret_cast<uint64_t>(gvas[i]), sizes[i], info);
        if (findRet == MMC_WRITE_READABLE_BLOB) {
            continue;
        }
        if (findRet != MMC_OK) {
            MMC_LOG_ERROR("client " << name_ << " batch copy write prepare local gva info failed, gva:"
                                    << reinterpret_cast<uint64_t>(gvas[i]) << ", size:" << sizes[i]
                                    << ", ret:" << findRet);
            return findRet;
        }
        toGvas.push_back(gvas[i]);
        toBuffers.push_back(buffers[i]);
        toSizes.push_back(sizes[i]);
    }
    if (toGvas.empty()) {
        return MMC_OK;
    }
    MMC_LOG_DEBUG("client " << name_ << " batch copy write path, count=" << gvas.size() << ", direct=" << direct
                            << ", state flip deferred to BatchWriteFinish");
    return BatchDataOperation(toGvas, toBuffers, toSizes, direct);
}

Result MmcClientDefault::BatchCopyReadPath(std::vector<void *> &gvas, std::vector<void *> &buffers,
                                           std::vector<size_t> &sizes, int32_t direct)
{
    std::vector<LocalGvaBlobInfo> readInfos;
    readInfos.reserve(gvas.size());
    for (size_t i = 0; i < gvas.size(); ++i) {
        LocalGvaBlobInfo info{};
        Result findRet = gvaBlobTracker_.FindReadable(reinterpret_cast<uint64_t>(gvas[i]), sizes[i], info);
        if (findRet != MMC_OK) {
            MMC_LOG_ERROR("client " << name_ << " batch copy read prepare local gva info failed, gva:"
                                    << reinterpret_cast<uint64_t>(gvas[i]) << ", size:" << sizes[i]
                                    << ", ret:" << findRet);
            return findRet;
        }
        readInfos.push_back(info);
    }

    Result readResult = BatchDataOperation(gvas, buffers, sizes, direct);
    if (readResult != MMC_OK) {
        return readResult;
    }

    const uint64_t nowMs = NowMs();
    for (const auto &info : readInfos) {
        if (info.IsLeaseExpired(nowMs)) {
            MMC_LOG_ERROR("client " << name_ << " batch copy read lease expired.");
            return MMC_LEASE_EXPIRED;
        }
    }
    return MMC_OK;
}

Result MmcClientDefault::BatchCopy(std::vector<void *> &gvas, std::vector<void *> &buffers, std::vector<size_t> &sizes,
                                   const int32_t direct)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);

    if (sizes.empty()) {
        return MMC_OK;
    }

    if (direct == SMEMB_COPY_L2G || direct == SMEMB_COPY_H2G) {
        // Write 方向
        return BatchCopyWritePath(gvas, buffers, sizes, direct);
    }
    if (direct == SMEMB_COPY_G2L || direct == SMEMB_COPY_G2H) {
        // Read 方向
        return BatchCopyReadPath(gvas, buffers, sizes, direct);
    }

    MMC_LOG_ERROR("Invalid direct " << direct << " for batch copy, count:" << sizes.size());
    return MMC_ERROR;
}

Result MmcClientDefault::BatchWriteFinish(const std::vector<std::string> &keys,
                                          const std::vector<int32_t> &writeResults, std::vector<int32_t> &outResults)
{
    MMC_VALIDATE_RETURN(bmProxy_ != nullptr, "BmProxy is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(metaNetClient_ != nullptr, "MetaNetClient is null", MMC_CLIENT_NOT_INIT);
    MMC_VALIDATE_RETURN(!keys.empty(), "keys is empty", MMC_INVALID_PARAM);
    MMC_VALIDATE_RETURN(keys.size() == writeResults.size(),
                        "keys size (" << keys.size() << ") != writeResults size (" << writeResults.size() << ")",
                        MMC_INVALID_PARAM);

    outResults.assign(keys.size(), MMC_INVALID_PARAM);

    BatchUpdateRequest updateRequest{};
    std::vector<size_t> sentIdx;
    sentIdx.reserve(keys.size());
    updateRequest.keys_.reserve(keys.size());
    updateRequest.ranks_.reserve(keys.size());
    updateRequest.mediaTypes_.reserve(keys.size());
    updateRequest.actionResults_.reserve(keys.size());
    updateRequest.operateIds_.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        LocalGvaBlobInfo info{};
        Result findRet = gvaBlobTracker_.FindBlobByKey(keys[i], info);
        if (findRet != MMC_OK) {
            MMC_LOG_DEBUG("key " << keys[i] << " not found locally, because of lease timeout, skip");
            continue;
        }
        auto opId = gvaBlobTracker_.ReleaseLease(keys[i]);
        if (opId == UINT64_MAX) {
            MMC_LOG_DEBUG("key " << keys[i] << " not found locally, because of lease timeout, skip");
        }
        TP_TRACE_RECORD(TP_MMC_TRACKER_REMOVE_LEASE_FROM_WRITE, 1000ULL, opId == UINT64_MAX ? -1 : 0);
        updateRequest.keys_.push_back(keys[i]);
        updateRequest.ranks_.push_back(info.blob.rank_);
        updateRequest.mediaTypes_.push_back(info.blob.mediaType_);
        BlobActionResult action = writeResults[i] == MMC_OK ? MMC_WRITE_OK : MMC_WRITE_FAIL;
        updateRequest.actionResults_.push_back(action);
        updateRequest.operateIds_.push_back(opId);
        sentIdx.push_back(i);
    }

    if (updateRequest.keys_.empty()) {
        MMC_LOG_DEBUG("client " << name_ << " batch write finish: no valid keys to send");
        return MMC_OK;
    }

    TP_TRACE_BEGIN(TP_MMC_LOCAL_BATCH_UPDATE);
    BatchUpdateResponse updateResponse;
    Result updateResult = metaNetClient_->SyncCall(updateRequest, updateResponse, rpcRetryTimeOut_);
    TP_TRACE_END(TP_MMC_LOCAL_BATCH_UPDATE, updateResult);
    for (size_t i = 0; i < sentIdx.size(); ++i) {
        size_t keyIdx = sentIdx[i];
        if (writeResults[keyIdx] == MMC_OK) {
            gvaBlobTracker_.MarkWriteSuccess(keys[keyIdx]);
        }
    }
    if (updateResult != MMC_OK || updateResponse.results_.size() != sentIdx.size()) {
        MMC_LOG_ERROR("client " << name_ << " batch write finish failed:" << updateResult << ", sent=" << sentIdx.size()
                                << ", retSize=" << updateResponse.results_.size());
        return MMC_ERROR;
    }
    for (size_t i = 0; i < sentIdx.size(); ++i) {
        size_t keyIdx = sentIdx[i];
        outResults[keyIdx] = updateResponse.results_[i];
    }
    MMC_LOG_DEBUG("client " << name_ << " batch write finish done: keysCnt=" << keys.size()
                            << ", sent=" << sentIdx.size());
    return MMC_OK;
}

Result MmcClientDefault::BatchDataOperation(std::vector<void *> &gvas, std::vector<void *> &buffers,
                                            std::vector<size_t> &sizes, int32_t direct)
{
    size_t minBytesForConcurrency = batchChunkSize_ * batchChunkCount_;

    const bool isPut = (direct == SMEMB_COPY_L2G || direct == SMEMB_COPY_H2G);
    const MediaType mediaType = (direct == SMEMB_COPY_L2G || direct == SMEMB_COPY_G2L) ? MEDIA_HBM : MEDIA_DRAM;

    // 计算总数据量
    size_t totalSize = 0;
    for (size_t s : sizes) {
        totalSize += s;
    }

    // 小数据量：直接一次性调用，不切片、不并发
    if (totalSize <= minBytesForConcurrency || sizes.size() <= batchChunkCount_) {
        Result ret = isPut ? bmProxy_->BatchDataPut(buffers, gvas, sizes, mediaType)
                           : bmProxy_->BatchDataGet(gvas, buffers, sizes, mediaType);
        if (ret != MMC_OK) {
            MMC_LOG_ERROR((isPut ? "BatchDataPut" : "BatchDataGet")
                          << " (small data, direct call) failed, direct=" << direct << ", ret=" << ret);
        }
        return ret;
    }

    // 大数据量：分片 + 并发执行
    return ExecuteConcurrently(gvas, buffers, sizes, isPut, mediaType, batchChunkSize_);
}

Result MmcClientDefault::ExecuteConcurrently(const std::vector<void *> &gvas, const std::vector<void *> &buffers,
                                             const std::vector<size_t> &sizes, bool isPut, MediaType mediaType,
                                             size_t chunkSize)
{
    const size_t total = sizes.size();
    std::vector<std::future<Result>> futures;
    futures.reserve(total);

    size_t idx = 0;
    while (idx < total) {
        std::vector<void *> subGvas;
        std::vector<void *> subBuffers;
        std::vector<size_t> subSizes;

        size_t currentBytes = 0;
        while (idx < total && currentBytes < chunkSize) {
            const size_t thisSize = sizes[idx];

            subGvas.push_back(gvas[idx]);
            subBuffers.push_back(buffers[idx]);
            subSizes.push_back(thisSize);

            currentBytes += thisSize;
            ++idx;
        }

        if (subSizes.empty()) {
            break;
        }

        // 提交任务到线程池
        auto task = [this, sub_b = std::move(subBuffers), sub_g = std::move(subGvas), sub_s = std::move(subSizes),
                     mediaType, isPut]() mutable -> Result {
            return isPut ? bmProxy_->BatchDataPut(sub_b, sub_g, sub_s, mediaType)
                         : bmProxy_->BatchDataGet(sub_g, sub_b, sub_s, mediaType);
        };

        if (isPut) {
            futures.emplace_back(writeThreadPool_->Enqueue(std::move(task)));
        } else {
            futures.emplace_back(readThreadPool_->Enqueue(std::move(task)));
        }
    }

    // 等待所有并发任务完成
    Result finalResult = MMC_OK;
    for (auto &fut : futures) {
        Result r = fut.get();
        if (r != MMC_OK && finalResult == MMC_OK) {
            finalResult = r;
        }
    }

    if (finalResult != MMC_OK) {
        MMC_LOG_ERROR((isPut ? "BatchDataPut" : "BatchDataGet") << " (concurrent) failed, ret=" << finalResult);
    }
    return finalResult;
}

} // namespace mmc
} // namespace ock
