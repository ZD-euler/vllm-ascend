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

#include <iostream>
#include <limits>
#include <stdexcept>
#include <unistd.h>

#include "mmc_client.h"
#include "mmc_client_default.h"
#include "mmc.h"
#include "mmc_logger.h"
#include "mmc_types.h"
#include "mmc_ptracer.h"
#include "mmc_locality_strategy.h"
#include "smem_bm_def.h"
#include "mmcache_store.h"

namespace ock {
namespace mmc {

constexpr int MAX_LAYER_NUM = 255;   // 分层场景：传统神经网络模型
constexpr int MAX_BUFFER_NUM = 8192; // 多 buffer 场景：支持稀疏数据、分段存储等
constexpr int MAX_KEY_LEN = 256;
constexpr uint64_t MMC_DEVICE_VA_START = 0x100000000000UL; // NPU上的地址空间起始: 16T
constexpr uint64_t MMC_DEVICE_VA_SIZE = 0x80000000000UL;   // NPU上的地址空间范围: 8T

static bool CopyPutOptions(const ReplicateConfig &replicateConfig, mmc_put_options &options)
{
    if (replicateConfig.preferredLocalServiceIDs.size() > MAX_BLOB_COPIES) {
        MMC_LOG_ERROR("vector size is " << replicateConfig.preferredLocalServiceIDs.size()
                                        << ", Maximum number of copies is " << MAX_BLOB_COPIES);
        return false;
    }
    if (replicateConfig.replicaNum > MAX_BLOB_COPIES) {
        MMC_LOG_ERROR("replica number " << replicateConfig.replicaNum << " exceeds maximum number of copies ("
                                        << MAX_BLOB_COPIES << ")");
        return false;
    }
    if (replicateConfig.replicaNum == 0) {
        MMC_LOG_ERROR("replica number cannot be 0");
        return false;
    }
    options.mediaType = 0; // will set by client proxy
    options.policy = NATIVE_AFFINITY;
    std::fill(std::begin(options.preferredLocalServiceIDs), std::end(options.preferredLocalServiceIDs), -1);
    std::copy(std::begin(replicateConfig.preferredLocalServiceIDs), std::end(replicateConfig.preferredLocalServiceIDs),
              std::begin(options.preferredLocalServiceIDs));
    options.replicaNum = replicateConfig.replicaNum;
    return true;
}

SignalBlocker::SignalBlocker()
{
    sigemptyset(&set_);
    sigaddset(&set_, SIGINT);
    sigaddset(&set_, SIGTERM);
    sigaddset(&set_, SIGHUP);
    pthread_sigmask(SIG_BLOCK, &set_, &oldset_);
}

SignalBlocker::~SignalBlocker()
{
    pthread_sigmask(SIG_SETMASK, &oldset_, nullptr);
}

// ResourceTracker implementation using singleton pattern
ResourceTracker &ResourceTracker::getInstance()
{
    static ResourceTracker instance;
    return instance;
}

ResourceTracker::ResourceTracker()
{
    // Set up signal handlers
    struct sigaction sa {};
    sa.sa_handler = signalHandler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;

    // Register for common termination signals
    sigaction(SIGINT, &sa, nullptr);  // Ctrl+C
    sigaction(SIGTERM, &sa, nullptr); // kill command
    sigaction(SIGHUP, &sa, nullptr);  // Terminal closed

    // Register exit handler
    std::atexit(exitHandler);
}

ResourceTracker::~ResourceTracker()
{
    // Cleanup is handled by exitHandler
}

void ResourceTracker::registerInstance(MmcacheStore *instance)
{
    SignalBlocker blocker;
    std::lock_guard<std::mutex> lock(mutex_);
    instances_.insert(instance);
}

void ResourceTracker::unregisterInstance(MmcacheStore *instance)
{
    SignalBlocker blocker;
    std::lock_guard<std::mutex> lock(mutex_);
    instances_.erase(instance);
}

void ResourceTracker::cleanupAllResources()
{
    SignalBlocker blocker;
    std::lock_guard<std::mutex> lock(mutex_);

    // Perform cleanup outside the lock to avoid potential deadlocks
    for (void *instance : instances_) {
        auto *store = static_cast<MmcacheStore *>(instance);
        if (store) {
            std::cout << "Cleaning up MmcacheStore instance" << std::endl;
            store->TearDown();
        }
    }
}

void ResourceTracker::signalHandler(int signal)
{
    const char msg[] = "Received signal, cleaning up resources\n";
    if (write(STDERR_FILENO, msg, sizeof(msg) - 1) < 0) {}

    getInstance().cleanupAllResources();

    // Re-raise the signal with default handler to allow normal termination
    struct sigaction sa {};
    sa.sa_handler = SIG_DFL;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(signal, &sa, nullptr);
    raise(signal);
}

void ResourceTracker::exitHandler()
{
    getInstance().cleanupAllResources();
}

MmcacheStore::MmcacheStore()
{
    // Register this instance with the global tracker
    ResourceTracker::getInstance().registerInstance(this);
}

MmcacheStore::~MmcacheStore()
{
    // Unregister from the tracker before cleanup
    ResourceTracker::getInstance().unregisterInstance(this);
}

ObjectStore::~ObjectStore() {}

std::shared_ptr<ObjectStore> ObjectStore::CreateObjectStore()
{
    return std::make_shared<MmcacheStore>();
}

int MmcacheStore::Setup(const local_config &config)
{
    return mmc_setup(&config);
}

int MmcacheStore::Init(const uint32_t deviceId, const bool initBm)
{
    mmc_init_config config{deviceId, initBm};
    return mmc_init(&config);
}

int MmcacheStore::TearDown()
{
    mmc_uninit();
    return 0;
}

int MmcacheStore::RegisterBuffer(void *buffer, size_t size)
{
    return mmcc_register_buffer(reinterpret_cast<uint64_t>(buffer), size);
}

int MmcacheStore::UnRegisterBuffer(void *buffer, size_t size)
{
    return mmcc_unregister_buffer(reinterpret_cast<uint64_t>(buffer), size);
}

int MmcacheStore::GetInto(const std::string &key, void *buffer, size_t size, const int32_t direct)
{
    uint32_t type = 0;
    switch (direct) {
        case SMEMB_COPY_G2L:
            type = MEDIA_HBM;
            break;
        case SMEMB_COPY_G2H:
            type = MEDIA_DRAM;
            break;
        default:
            MMC_LOG_ERROR("Failed to get by type " << direct << " for key " << key);
            return -1;
    }
    mmc_buffer mmcBuffer = {.addr = reinterpret_cast<uint64_t>(buffer), .type = type, .offset = 0, .len = size};
    TP_TRACE_BEGIN(TP_MMC_PY_GET);
    auto res = mmcc_get(key.c_str(), &mmcBuffer, 0);
    TP_TRACE_END(TP_MMC_PY_GET, res);
    if (res != MMC_OK) {
        MMC_LOG_ERROR("Failed to get key " << key << ", error code: " << res);
    }
    return res;
}

int MmcacheStore::GetLocalServiceId(uint32_t &localServiceId)
{
    return mmcc_local_service_id(&localServiceId);
}

int MmcacheStore::PutFrom(const std::string &key, void *buffer, size_t size, const int32_t direct,
                          const ReplicateConfig &replicateConfig)
{
    uint32_t type = 0;
    switch (direct) {
        case SMEMB_COPY_L2G:
            type = MEDIA_HBM;
            break;
        case SMEMB_COPY_H2G:
            type = MEDIA_DRAM;
            break;
        default:
            MMC_LOG_ERROR("Failed to put by type " << direct << " for key " << key);
            return -1;
    }
    mmc_buffer mmcBuffer = {.addr = reinterpret_cast<uint64_t>(buffer), .type = type, .offset = 0, .len = size};

    mmc_put_options options{};
    auto temp = CopyPutOptions(replicateConfig, options);
    MMC_ASSERT_LOG_AND_RETURN(temp, "CopyPutOptions(replicateConfig, options) = " << temp, MMC_ERROR);
    TP_TRACE_BEGIN(TP_MMC_PY_PUT);
    const auto res = mmcc_put(key.c_str(), &mmcBuffer, options, 0);
    auto ret = ReturnWrapper(res, key);
    TP_TRACE_END(TP_MMC_PY_PUT, ret);
    return ret;
}

int MmcacheStore::Remove(const std::string &key)
{
    TP_TRACE_BEGIN(TP_MMC_PY_REMOVE);
    auto ret = mmcc_remove(key.c_str(), 0); // 0 - success, other - not success
    TP_TRACE_END(TP_MMC_PY_REMOVE, ret);
    return ret;
}

std::vector<int> MmcacheStore::BatchRemove(const std::vector<std::string> &keys)
{
    std::vector<int> results;

    if (keys.empty()) {
        MMC_LOG_DEBUG("key vector is empty");
        return {};
    }

    results.resize(keys.size(), -1);
    const char **c_keys = new (std::nothrow) const char *[keys.size()];
    if (c_keys == nullptr) {
        MMC_LOG_ERROR("Cannot malloc memory for keys! keys.size=" << keys.size());
        return results; // Return vector filled with error code
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        c_keys[i] = keys[i].c_str();
    }
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_REMOVE);
    int32_t res = mmcc_batch_remove(c_keys, keys.size(), results.data(), 0);
    TP_TRACE_END(TP_MMC_PY_BATCH_REMOVE, res);
    if (res != 0) {
        MMC_LOG_ERROR("remove_batch failed, res=" << res << ", keys.size=" << keys.size());
        std::fill(results.begin(), results.end(), res);
        delete[] c_keys;
        return results; // Return vector filled with error code
    }

    delete[] c_keys;
    return results;
}

int MmcacheStore::RemoveAll()
{
    MMC_VALIDATE_RETURN(MmcClientDefault::GetInstance() != nullptr, "client is not initialize", MMC_CLIENT_NOT_INIT);
    MMC_RETURN_ERROR(MmcClientDefault::GetInstance()->RemoveAll(0), MmcClientDefault::GetInstance()->Name()
                                                                        << " remove all keys failed!");
    return MMC_OK;
}

int MmcacheStore::IsExist(const std::string &key)
{
    TP_TRACE_BEGIN(TP_MMC_PY_EXIST);
    int32_t res = mmcc_exist(key.c_str(), 0);
    TP_TRACE_END(TP_MMC_PY_EXIST, res);
    if (res == MMC_OK) {
        // align with mooncake: 1 represents exist
        return 1;
    } else if (res == MMC_UNMATCHED_KEY) {
        // align with mooncake: 0 represents not exist
        return 0;
    }
    return res;
}

std::vector<int> MmcacheStore::BatchIsExist(const std::vector<std::string> &keys)
{
    std::vector<int> results;

    if (keys.empty()) {
        MMC_LOG_DEBUG("key vector is empty");
        return {};
    }

    results.resize(keys.size(), -1);
    const char **c_keys = new (std::nothrow) const char *[keys.size()];
    if (c_keys == nullptr) {
        MMC_LOG_ERROR("Cannot malloc memory for keys! keys.size=" << keys.size());
        return results; // Return vector filled with error code -1
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        c_keys[i] = keys[i].c_str();
    }
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_EXIST);
    int32_t res = mmcc_batch_exist(c_keys, keys.size(), results.data(), 0);
    TP_TRACE_END(TP_MMC_PY_BATCH_EXIST, res);
    if (res != 0) {
        MMC_LOG_ERROR("batch_exist failed:" << res);
        std::fill(results.begin(), results.end(), res);
        delete[] c_keys;
        return results; // Return vector filled with error code
    }

    for (int &result : results) {
        if (result == MMC_OK) {
            // align with mooncake: 1 represents exist
            result = 1;
        } else if (result == MMC_UNMATCHED_KEY) {
            // align with mooncake: 0 represents not exist
            result = 0;
        }
    }

    delete[] c_keys;
    return results;
}

KeyInfo MmcacheStore::GetKeyInfo(const std::string &key, uint32_t flag)
{
    mmc_data_info info;
    TP_TRACE_BEGIN(TP_MMC_PY_QUERY);
    auto res = mmcc_query(key.c_str(), &info, flag);
    TP_TRACE_END(TP_MMC_PY_QUERY, res);
    if (res != MMC_OK) {
        MMC_LOG_ERROR("Failed to query key " << key << ", error code: " << res);
        return {0, 0};
    }

    if (!info.valid) {
        MMC_LOG_ERROR("Failed to query key " << key << ", info invalid");
        return {0, 0};
    }

    KeyInfo keyInfo{info.size, info.numBlobs};
    for (int i = 0; i < info.numBlobs; i++) {
        keyInfo.AddLoc(info.ranks[i]);
        keyInfo.AddType(info.types[i]);
        keyInfo.AddGva(info.gvas[i]);
    }
    return keyInfo;
}

std::vector<KeyInfo> MmcacheStore::BatchGetKeyInfo(const std::vector<std::string> &keys, uint32_t flag)
{
    uint32_t size = keys.size();

    if (keys.empty()) {
        MMC_LOG_DEBUG("key vector is empty");
        return {};
    }

    const char **ckeys = new (std::nothrow) const char *[size];
    if (ckeys == nullptr) {
        MMC_LOG_ERROR("Cannot malloc memory for keys! keys.size=" << keys.size());
        return {};
    }
    for (uint32_t i = 0; i < size; ++i) {
        ckeys[i] = keys[i].c_str();
    }

    mmc_data_info *infoArr = (mmc_data_info *)malloc(size * sizeof(mmc_data_info));
    if (infoArr == nullptr) {
        delete[] ckeys;
        MMC_LOG_ERROR("Cannot malloc memory for infos!");
        return {};
    }
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_QUERY);
    auto ret = mmcc_batch_query(ckeys, size, infoArr, flag);
    TP_TRACE_END(TP_MMC_PY_BATCH_QUERY, ret);
    if (ret != MMC_OK) {
        delete[] ckeys;
        free(infoArr);
        MMC_LOG_ERROR("batch query failed! ret:" << ret);
        return {};
    }

    std::vector<KeyInfo> infoList{};
    for (uint32_t i = 0; i < size; ++i) {
        mmc_data_info &info = infoArr[i];
        if (!info.valid) {
            infoList.emplace_back(KeyInfo{0, 0});
            continue;
        }

        KeyInfo keyInfo{info.size, info.numBlobs};
        for (int j = 0; j < info.numBlobs && j < MAX_BLOB_COPIES; j++) {
            keyInfo.AddLoc(info.ranks[j]);
            keyInfo.AddType(info.types[j]);
            keyInfo.AddGva(info.gvas[j]);
        }
        infoList.emplace_back(keyInfo);
    }

    delete[] ckeys;
    free(infoArr);
    return infoList;
}

int MmcacheStore::PrefetchKeys(std::vector<std::string> keys, uint8_t srcMedia, uint8_t dstMedia, uint32_t flags)
{
    if (keys.empty()) {
        MMC_LOG_DEBUG("key vector is empty");
        return MMC_INVALID_PARAM;
    }

    const char **ckeys = new (std::nothrow) const char *[keys.size()];
    if (ckeys == nullptr) {
        MMC_LOG_ERROR("Cannot malloc memory for keys! keys.size=" << keys.size());
        return MMC_MALLOC_FAILED;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        ckeys[i] = keys[i].c_str();
    }

    auto ret = mmcc_prefetch_keys(ckeys, keys.size(), srcMedia, dstMedia, flags);
    delete[] ckeys;
    return ret;
}

int MmcacheStore::EvictKeys(std::vector<std::string> keys, uint8_t srcMedia, uint8_t dstMedia, uint32_t flags)
{
    if (keys.empty()) {
        MMC_LOG_DEBUG("key vector is empty");
        return MMC_INVALID_PARAM;
    }

    const char **ckeys = new (std::nothrow) const char *[keys.size()];
    if (ckeys == nullptr) {
        MMC_LOG_ERROR("Cannot malloc memory for keys! keys.size=" << keys.size());
        return MMC_MALLOC_FAILED;
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        ckeys[i] = keys[i].c_str();
    }

    auto ret = mmcc_evict_keys(ckeys, keys.size(), srcMedia, dstMedia, flags);
    delete[] ckeys;
    return ret;
}

std::vector<int> MmcacheStore::BatchPutFrom(const std::vector<std::string> &keys, const std::vector<void *> &buffers,
                                            const std::vector<size_t> &sizes, const int32_t direct,
                                            const ReplicateConfig &replicateConfig)
{
    const size_t count = keys.size();
    MMC_VALIDATE_RETURN(count > 0, "key vector is empty", {});

    std::vector<int> results(count, MMC_INVALID_PARAM);
    if (buffers.size() != count || sizes.size() != count) {
        MMC_LOG_ERROR("Input vector sizes mismatch: keys=" << keys.size() << ", buffers=" << buffers.size()
                                                           << ", sizes=" << sizes.size());
        return results;
    }
    uint32_t type = 0;
    switch (direct) {
        case SMEMB_COPY_L2G:
            type = MEDIA_HBM;
            break;
        case SMEMB_COPY_H2G:
            type = MEDIA_DRAM;
            break;
        default:
            MMC_LOG_ERROR("Failed to batch put by type " << direct);
            return results;
    }

    std::vector<const char *> keyArray(count);
    std::vector<mmc_buffer> bufferArray(count);
    for (size_t i = 0; i < count; ++i) {
        keyArray[i] = keys[i].c_str();
        bufferArray[i] = {.addr = reinterpret_cast<uint64_t>(buffers[i]),
                          .type = type,
                          .offset = 0,
                          .len = static_cast<uint64_t>(sizes[i])};
    }

    mmc_put_options options{};
    auto temp = CopyPutOptions(replicateConfig, options);
    MMC_ASSERT_LOG_AND_RETURN(temp, "CopyPutOptions(replicateConfig, options) = " << temp, results);
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_PUT);
    mmcc_batch_put(keyArray.data(), count, bufferArray.data(), options, ALLOC_RANDOM, results.data());
    TP_TRACE_END(TP_MMC_PY_BATCH_PUT, 0);
    for (size_t i = 0; i < count; i++) {
        results[i] = ReturnWrapper(results[i], keys[i]);
    }
    return results;
}

std::vector<int> MmcacheStore::BatchGetInto(const std::vector<std::string> &keys, const std::vector<void *> &buffers,
                                            const std::vector<size_t> &sizes, const int32_t direct)
{
    size_t count = keys.size();
    MMC_VALIDATE_RETURN(count > 0, "key vector is empty", {});

    std::vector<int> results(count, MMC_INVALID_PARAM);
    if (buffers.size() != count || sizes.size() != count) {
        MMC_LOG_ERROR("Input vector sizes mismatch: keys=" << keys.size() << ", buffers=" << buffers.size()
                                                           << ", sizes=" << sizes.size());
        return results;
    }
    uint32_t type = 0;
    switch (direct) {
        case SMEMB_COPY_G2L:
            type = MEDIA_HBM;
            break;
        case SMEMB_COPY_G2H:
            type = MEDIA_DRAM;
            break;
        default:
            MMC_LOG_ERROR("Failed to batch get by type " << direct);
            return results;
    }

    std::vector<const char *> keyArray(count);
    std::vector<mmc_buffer> bufferArray(count);
    for (size_t i = 0; i < count; ++i) {
        if (static_cast<uint64_t>(sizes[i]) == 0) {
            MMC_LOG_ERROR("Invalid vector sizes on idx [" << i << "] equals 0");
            return results;
        }
        keyArray[i] = keys[i].c_str();
        bufferArray[i] = {.addr = reinterpret_cast<uint64_t>(buffers[i]),
                          .type = type,
                          .offset = 0,
                          .len = static_cast<uint64_t>(sizes[i])};
    }
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_GET);
    auto ret = mmcc_batch_get(keyArray.data(), count, bufferArray.data(), 0, results.data());
    TP_TRACE_END(TP_MMC_PY_BATCH_GET, ret);
    (void)ret;
    return results;
}

bool MmcacheStore::IsInHybmDeviceRange(uint64_t va)
{
    return (va >= MMC_DEVICE_VA_START) && (va < (MMC_DEVICE_VA_START + MMC_DEVICE_VA_SIZE));
}

int MmcacheStore::PutFromLayers(const std::string &key, const std::vector<void *> &buffers,
                                const std::vector<size_t> &sizes, const int32_t direct,
                                const ReplicateConfig &replicateConfig)
{
    MMC_ASSERT_LOG_AND_RETURN(MmcClientDefault::GetInstance() != nullptr, "MmcClientDefault::GetInstance() is nullptr",
                              MMC_INVALID_PARAM);
    if (direct != SMEMB_COPY_L2G && direct != SMEMB_COPY_H2G && direct != SMEMB_COPY_AUTO) {
        MMC_LOG_ERROR(
            "Invalid direct(" << direct
                              << "), only 0 (SMEMB_COPY_L2G), 3 (SMEMB_COPY_H2G) and 9 (SMEMB_COPY_AUTO) is supported");
        return MMC_INVALID_PARAM;
    }

    uint32_t type = MEDIA_DRAM;
    if (direct == SMEMB_COPY_L2G) {
        type = MEDIA_HBM;
    } else if (direct == SMEMB_COPY_H2G) {
        type = MEDIA_DRAM;
    } else if (direct == SMEMB_COPY_AUTO) {
        MMC_ASSERT_LOG_AND_RETURN(!buffers.empty(), "buffers is empty", MMC_INVALID_PARAM);
        uint64_t va = reinterpret_cast<uint64_t>(buffers[0]);
        type = IsInHybmDeviceRange(va) ? MEDIA_HBM : MEDIA_DRAM;
    }

    if (key.length() == 0 || key.length() > MAX_KEY_LEN) {
        MMC_LOG_ERROR("Invalid param, key's len (" << key.length() << ") is not between 1 and " << MAX_KEY_LEN);
        return MMC_INVALID_PARAM;
    }

    auto layerNum = buffers.size();
    if (layerNum == 0 || layerNum > MAX_LAYER_NUM) {
        MMC_LOG_ERROR("Layer number is 0 or exceeds the limit of " << MAX_LAYER_NUM << ", layerNum=" << layerNum);
        return MMC_INVALID_PARAM;
    }

    if (sizes.size() != layerNum) {
        MMC_LOG_ERROR("Unmatched number of layers:" << layerNum << " and sizes:" << sizes.size());
        return MMC_INVALID_PARAM;
    }

    mmc_put_options options{};
    auto temp = CopyPutOptions(replicateConfig, options);
    MMC_ASSERT_LOG_AND_RETURN(temp, "CopyPutOptions(replicateConfig, options) = " << temp, MMC_ERROR);
    Result res;
    MmcBufferArray bufArr;
    for (size_t i = 0; i < layerNum; i += 1) {
        if (static_cast<uint64_t>(sizes[i]) == 0) {
            MMC_LOG_ERROR("Invalid param, layers buf len on idx [" << i << "] equals 0");
            return MMC_INVALID_PARAM;
        }
        bufArr.AddBuffer({.addr = reinterpret_cast<uint64_t>(buffers[i]),
                          .type = type,
                          .offset = 0,
                          .len = static_cast<uint64_t>(sizes[i])});
    }
    TP_TRACE_BEGIN(TP_MMC_PY_PUT_LAYERS);
    res = MmcClientDefault::GetInstance()->Put(key, bufArr, options, 0);
    TP_TRACE_END(TP_MMC_PY_PUT_LAYERS, res);

    return ReturnWrapper(res, key);
}

std::vector<int> MmcacheStore::BatchPutFromLayers(const std::vector<std::string> &keys,
                                                  const std::vector<std::vector<void *>> &buffers,
                                                  const std::vector<std::vector<size_t>> &sizes, const int32_t direct,
                                                  const ReplicateConfig &replicateConfig)
{
    MMC_ASSERT_LOG_AND_RETURN(MmcClientDefault::GetInstance() != nullptr, "MmcClientDefault::GetInstance() is nullptr",
                              {});
    const size_t batchSize = keys.size();
    MMC_VALIDATE_RETURN(batchSize > 0, "key vector is empty", {});

    std::vector<int> results(batchSize, MMC_INVALID_PARAM);

    if (direct != SMEMB_COPY_L2G && direct != SMEMB_COPY_H2G && direct != SMEMB_COPY_AUTO) {
        MMC_LOG_ERROR(
            "Invalid direct(" << direct
                              << "), only"
                                 " 0 (SMEMB_COPY_L2G) , 3 (SMEMB_COPY_H2G) and 9 (SMEMB_COPY_AUTO) is supported");
        return results;
    }

    uint32_t type = MEDIA_DRAM;
    if (direct == SMEMB_COPY_AUTO) {
        if (buffers.empty() || buffers[0].empty()) {
            MMC_LOG_ERROR("Buffers is empty, can't auto detect media type");
            return results;
        }
        uint64_t va = reinterpret_cast<uint64_t>(buffers[0][0]);
        type = IsInHybmDeviceRange(va) ? MEDIA_HBM : MEDIA_DRAM;
    } else {
        type = (direct == SMEMB_COPY_L2G ? MEDIA_HBM : MEDIA_DRAM);
    }

    if (batchSize != buffers.size() || batchSize != sizes.size()) {
        MMC_LOG_ERROR("Input vector sizes mismatch: keys=" << keys.size() << ", buffers=" << buffers.size()
                                                           << ", sizes=" << sizes.size());
        return results;
    }

    for (const std::string &key : keys) {
        if (key.length() == 0 || key.length() > MAX_KEY_LEN) {
            MMC_LOG_ERROR("Invalid param, key's len (" << key.length() << ") is not between 1 and " << MAX_KEY_LEN);
            return results;
        }
    }

    auto res = CheckInput(batchSize, buffers, sizes);
    if (res != MMC_OK) {
        MMC_LOG_ERROR("Failed to check if all layers are 2D, ret=" << res);
        return results;
    }

    mmc_put_options options{};
    auto temp = CopyPutOptions(replicateConfig, options);
    MMC_ASSERT_LOG_AND_RETURN(temp, "opyPutOptions(replicateConfig, options) = " << temp, results);
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_PUT_LAYERS);
    std::vector<MmcBufferArray> bufferArrays;
    GetBufferArrays(batchSize, type, buffers, sizes, bufferArrays);
    auto ret = MmcClientDefault::GetInstance()->BatchPut(keys, bufferArrays, options, ALLOC_RANDOM, results);
    TP_TRACE_END(TP_MMC_PY_BATCH_PUT_LAYERS, ret);
    (void)ret;
    for (size_t i = 0; i < batchSize; i++) {
        results[i] = ReturnWrapper(results[i], keys[i]);
    }

    return results;
}

int MmcacheStore::GetIntoLayers(const std::string &key, const std::vector<void *> &buffers,
                                const std::vector<size_t> &sizes, const int32_t direct)
{
    if (direct != SMEMB_COPY_G2L && direct != SMEMB_COPY_G2H && direct != SMEMB_COPY_AUTO) {
        MMC_LOG_ERROR(
            "Invalid direct(" << direct
                              << "), only"
                                 "1 (SMEMB_COPY_G2L) , 2 (SMEMB_COPY_G2H) and 9 (SMEMB_COPY_AUTO) is supported");
        return MMC_INVALID_PARAM;
    }
    MMC_ASSERT_LOG_AND_RETURN(MmcClientDefault::GetInstance() != nullptr, "MmcClientDefault::GetInstance() is nullptr",
                              MMC_INVALID_PARAM);

    uint32_t type = MEDIA_DRAM;
    if (direct == SMEMB_COPY_G2L) {
        type = MEDIA_HBM;
    } else if (direct == SMEMB_COPY_G2H) {
        type = MEDIA_DRAM;
    } else if (direct == SMEMB_COPY_AUTO) {
        MMC_ASSERT_LOG_AND_RETURN(!buffers.empty(), "buffers is empty", MMC_INVALID_PARAM);
        uint64_t va = reinterpret_cast<uint64_t>(buffers[0]);
        type = IsInHybmDeviceRange(va) ? MEDIA_HBM : MEDIA_DRAM;
    }

    if (key.length() == 0 || key.length() > MAX_KEY_LEN) {
        MMC_LOG_ERROR("Invalid param, key's len (" << key.length() << ") is not between 1 and " << MAX_KEY_LEN);
        return MMC_INVALID_PARAM;
    }

    auto layerNum = buffers.size();
    if (layerNum == 0 || layerNum > MAX_LAYER_NUM) {
        MMC_LOG_ERROR("Layer number is 0 or exceeds the limit of " << MAX_LAYER_NUM << ", layerNum=" << layerNum);
        return MMC_INVALID_PARAM;
    }

    if (sizes.size() != layerNum) {
        MMC_LOG_ERROR("Unmatched number of layers:" << layerNum << " and sizes:" << sizes.size());
        return MMC_INVALID_PARAM;
    }

    std::vector<mmc_buffer> mmc_buffers;
    for (size_t i = 0; i < layerNum; i += 1) {
        if (static_cast<uint64_t>(sizes[i]) == 0) {
            MMC_LOG_ERROR("Invalid param, layers buf len on idx [" << i << "] equals 0");
            return MMC_INVALID_PARAM;
        }
        mmc_buffers.push_back({.addr = reinterpret_cast<uint64_t>(buffers[i]),
                               .type = type,
                               .offset = 0,
                               .len = static_cast<uint64_t>(sizes[i])});
    }
    MmcBufferArray bufArr(mmc_buffers);
    TP_TRACE_BEGIN(TP_MMC_PY_GET_LAYERS);
    auto ret = MmcClientDefault::GetInstance()->Get(key, bufArr, 0);
    TP_TRACE_END(TP_MMC_PY_GET_LAYERS, ret);
    return ret;
}

std::vector<int> MmcacheStore::BatchGetIntoLayers(const std::vector<std::string> &keys,
                                                  const std::vector<std::vector<void *>> &buffers,
                                                  const std::vector<std::vector<size_t>> &sizes, const int32_t direct)
{
    MMC_ASSERT_LOG_AND_RETURN(MmcClientDefault::GetInstance() != nullptr, "MmcClientDefault::GetInstance() is nullptr",
                              {});
    const size_t batchSize = keys.size();
    MMC_VALIDATE_RETURN(batchSize > 0, "key vector is empty", {});

    std::vector<int> results(batchSize, MMC_INVALID_PARAM);

    if (direct != SMEMB_COPY_G2L && direct != SMEMB_COPY_G2H && direct != SMEMB_COPY_AUTO) {
        MMC_LOG_ERROR(
            "Invalid direct(" << direct
                              << "), only"
                                 " 1 (SMEMB_COPY_G2L) , 2 (SMEMB_COPY_G2H) and 9 (SMEMB_COPY_AUTO) is supported");
        return results;
    }
    uint32_t type = MEDIA_DRAM;
    if (direct == SMEMB_COPY_AUTO) {
        if (buffers.empty() || buffers[0].empty()) {
            MMC_LOG_ERROR("Buffers empty, cannot infer memory type for SMEMB_COPY_AUTO");
            return results;
        }
        uint64_t va = reinterpret_cast<uint64_t>(buffers[0][0]);
        type = IsInHybmDeviceRange(va) ? MEDIA_HBM : MEDIA_DRAM;
    } else {
        type = (direct == SMEMB_COPY_G2L ? MEDIA_HBM : MEDIA_DRAM);
    }

    if (batchSize != buffers.size() || batchSize != sizes.size()) {
        MMC_LOG_ERROR("Input vector sizes mismatch: keys=" << keys.size() << ", buffers=" << buffers.size()
                                                           << ", sizes=" << sizes.size());
        return results;
    }

    for (const std::string &key : keys) {
        if (key.length() == 0 || key.length() > MAX_KEY_LEN) {
            MMC_LOG_ERROR("Invalid param, key's len (" << key.length() << ") is not between 1 and " << MAX_KEY_LEN);
            return results;
        }
    }

    auto res = CheckInput(batchSize, buffers, sizes);
    if (res != MMC_OK) {
        MMC_LOG_ERROR("Failed to check if all layers are 2D, ret=" << res);
        return results;
    }

    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_GET_LAYERS);
    std::vector<MmcBufferArray> bufferArrays;
    GetBufferArrays(batchSize, type, buffers, sizes, bufferArrays);
    auto ret = MmcClientDefault::GetInstance()->BatchGet(keys, bufferArrays, 0, results);
    TP_TRACE_END(TP_MMC_PY_BATCH_GET_LAYERS, ret);
    (void)ret;
    return results;
}

int MmcacheStore::CheckInput(const size_t batchSize, const std::vector<std::vector<void *>> &buffers,
                             const std::vector<std::vector<size_t>> &sizes)
{
    for (size_t i = 0; i < batchSize; i += 1) {
        const auto layerNum = buffers[i].size();
        if (layerNum == 0 || layerNum > MAX_BUFFER_NUM) {
            MMC_LOG_ERROR("Layer number is 0 or exceeds the limit of " << MAX_BUFFER_NUM << " for layerNum "
                                                                       << layerNum);
            return MMC_INVALID_PARAM;
        }
        if (sizes[i].size() != layerNum) {
            MMC_LOG_ERROR("Unmatched number of layers:" << layerNum << " and sizes:" << sizes[i].size());
            return MMC_INVALID_PARAM;
        }
        for (const auto &size : sizes[i]) {
            if (size <= 0) {
                MMC_LOG_ERROR("Invalid param, size should > 0");
                return MMC_INVALID_PARAM;
            }
        }
    }
    return MMC_OK;
}

void MmcacheStore::GetBufferArrays(const size_t batchSize, const uint32_t type,
                                   const std::vector<std::vector<void *>> &bufferLists,
                                   const std::vector<std::vector<size_t>> &sizeLists,
                                   std::vector<MmcBufferArray> &bufferArrays)
{
    for (size_t i = 0; i < batchSize; i += 1) {
        const auto &buffers = bufferLists[i];
        const auto &sizes = sizeLists[i];
        const auto layerNum = buffers.size();

        std::vector<mmc_buffer> mmc_buffers;
        for (size_t l = 0; l < layerNum; l += 1) {
            mmc_buffers.push_back(
                {.addr = reinterpret_cast<uint64_t>(buffers[l]), .type = type, .offset = 0, .len = sizes[l]});
        }
        MmcBufferArray bufArr(mmc_buffers);
        bufferArrays.push_back(bufArr);
    }
}

int MmcacheStore::ReturnWrapper(const int result, const std::string &key)
{
    if (result != MMC_OK) {
        if (result == MMC_DUPLICATED_OBJECT) {
            MMC_LOG_DEBUG("Duplicated key " << key << ", put operation skipped");
            return MMC_OK;
        } else {
            MMC_LOG_ERROR("Failed to put key " << key << ", error code=" << result);
            return result;
        }
    }
    return MMC_OK;
}

int MmcacheStore::Put(const std::string &key, mmc_buffer &buffer, const ReplicateConfig &replicateConfig)
{
    mmc_put_options options{};
    auto temp = CopyPutOptions(replicateConfig, options);
    MMC_ASSERT_LOG_AND_RETURN(temp, "CopyPutOptions(replicateConfig, options); = " << temp, MMC_ERROR);
    TP_TRACE_BEGIN(TP_MMC_PY_PUT);
    const auto res = mmcc_put(key.c_str(), &buffer, options, 0);
    auto ret = ReturnWrapper(res, key);
    TP_TRACE_END(TP_MMC_PY_PUT, ret);
    return ret;
}

int MmcacheStore::PutBatch(const std::vector<std::string> &keys, std::vector<mmc_buffer> &buffers,
                           const ReplicateConfig &replicateConfig)
{
    const size_t count = keys.size();
    MMC_VALIDATE_RETURN(count > 0, "key vector is empty", 0);

    std::vector<int> results(count, MMC_INVALID_PARAM);
    if (buffers.size() != count) {
        MMC_LOG_ERROR("Input vector sizes mismatch: keys=" << keys.size() << ", buffers=" << buffers.size());
        return MMC_INVALID_PARAM;
    }

    std::vector<const char *> keyArray(count);
    std::vector<mmc_buffer> bufferArray(count);
    for (size_t i = 0; i < count; ++i) {
        keyArray[i] = keys[i].c_str();
        bufferArray[i] = buffers[i];
    }

    mmc_put_options options{};
    auto temp = CopyPutOptions(replicateConfig, options);
    MMC_ASSERT_LOG_AND_RETURN(temp, "CopyPutOptions(replicateConfig, options) = " << temp, MMC_INVALID_PARAM);
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_PUT);
    mmcc_batch_put(keyArray.data(), count, bufferArray.data(), options, ALLOC_RANDOM, results.data());
    TP_TRACE_END(TP_MMC_PY_BATCH_PUT, 0);
    for (size_t i = 0; i < count; i++) {
        results[i] = ReturnWrapper(results[i], keys[i]);
        if (results[i] != MMC_OK) {
            return results[i];
        }
    }
    return MMC_OK;
}

mmc_buffer MmcacheStore::Get(const std::string &key)
{
    mmc_data_info info;
    auto res = mmcc_query(key.c_str(), &info, 0);
    if (res != MMC_OK) {
        MMC_LOG_ERROR("Failed to query key " << key << ", error code: " << res);
        return {};
    }

    if (!info.valid) {
        MMC_LOG_ERROR("Failed to query key " << key << ", info invalid");
        return {};
    }

    const auto dataPtr = new (std::nothrow) char[info.size];
    if (dataPtr == nullptr) {
        MMC_LOG_ERROR("Failed to allocate dynamic memory.");
        return {};
    }
    mmc_buffer buffer = {
        .addr = reinterpret_cast<uintptr_t>(dataPtr),
        .type = MEDIA_DRAM,
        .offset = 0,
        .len = info.size,
    };

    res = mmcc_get(key.c_str(), &buffer, 0);
    if (res != MMC_OK) {
        MMC_LOG_ERROR("Failed to get key " << key << ", error code: " << res);
        delete[] dataPtr;
        return {};
    }

    return buffer;
}

std::vector<mmc_buffer> MmcacheStore::GetBatch(const std::vector<std::string> &keys)
{
    size_t count = keys.size();
    MMC_VALIDATE_RETURN(count > 0, "key vector is empty", {});

    std::vector<int> results(count, MMC_INVALID_PARAM);
    std::vector<mmc_buffer> buffers(count, {0, 0, 0, 0});
    std::vector<const char *> keyArray(count);

    // 1. Query KeyInfo for all keys
    auto keyInfos = BatchGetKeyInfo(keys);
    if (keyInfos.size() != count) {
        MMC_LOG_ERROR("BatchGetKeyInfo returned " << keyInfos.size() << " results, expected " << count);
        return {};
    }

    // 2. alloc memory and assign value to the buffers
    for (size_t i = 0; i < count; ++i) {
        auto keyInfo = keyInfos[i];
        keyArray[i] = keys[i].c_str();
        const auto dataPtr = new (std::nothrow) char[keyInfo.Size()];
        if (dataPtr == nullptr) {
            // Release the newly allocated memory and set the addr of buffers to 0.
            for (size_t j = 0; j < i; ++j) {
                auto tmpPtr = reinterpret_cast<char *>(buffers[j].addr);
                delete[] tmpPtr;
                buffers[j].addr = 0;
                buffers[j].len = 0;
            }
            MMC_LOG_ERROR("Failed to allocate dynamic memory for key: " << keys[i].c_str());
            return {};
        }
        buffers[i] = {
            .addr = reinterpret_cast<uint64_t>(dataPtr),
            .type = MEDIA_DRAM,
            .offset = 0,
            .len = static_cast<uint64_t>(keyInfo.Size()),
        };
    }
    // 3. call batch get api
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_GET);
    auto ret = mmcc_batch_get(keyArray.data(), count, buffers.data(), 0, results.data());
    TP_TRACE_END(TP_MMC_PY_BATCH_GET, ret);
    if (ret != MMC_OK) {
        for (size_t i = 0; i < count; ++i) {
            auto tmpPtr = reinterpret_cast<char *>(buffers[i].addr);
            delete[] tmpPtr;
            buffers[i] = {0, 0, 0, 0};
        }
        MMC_LOG_ERROR("mmcc_batch_get failed, ret=" << ret);
        return {};
    }
    // 4. zero out buffers for individually failed keys
    for (size_t i = 0; i < count; ++i) {
        if (results[i] != MMC_OK) {
            auto tmpPtr = reinterpret_cast<char *>(buffers[i].addr);
            delete[] tmpPtr;
            buffers[i] = {0, 0, 0, 0};
        }
    }
    return buffers;
}

std::vector<int> MmcacheStore::BatchAddLease(const std::vector<std::string> &keys, uint64_t leaseTtlMs)
{
    if (keys.empty()) {
        MMC_LOG_ERROR("Invalid batch add lease input, key size:" << keys.size());
        return {};
    }
    std::vector<int> results(keys.size(), MMC_INVALID_PARAM);
    std::vector<const char *> keyArray(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keyArray[i] = keys[i].c_str();
    }
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_ADD_LEASE);
    Result ret = mmcc_batch_add_lease(keyArray.data(), keys.size(), leaseTtlMs, results.data());
    TP_TRACE_END(TP_MMC_PY_BATCH_ADD_LEASE, ret);
    return results;
}

std::pair<std::vector<int>, std::vector<KeyInfo>> MmcacheStore::BatchAddLeaseWithInfo(
    const std::vector<std::string> &keys, uint64_t leaseTtlMs)
{
    if (keys.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("too many keys for a lease batch");
    }
    std::vector<int> results(keys.size(), MMC_INVALID_PARAM);
    std::vector<KeyInfo> infos(keys.size(), KeyInfo{0, 0});
    if (keys.empty()) {
        return {std::move(results), std::move(infos)};
    }
    std::vector<std::string> validKeys;
    std::vector<size_t> validIndices;
    validKeys.reserve(keys.size());
    validIndices.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i].empty() || keys[i].size() > MAX_KEY_LEN) {
            continue;
        }
        validKeys.push_back(keys[i]);
        validIndices.push_back(i);
    }
    if (validKeys.empty()) {
        return {std::move(results), std::move(infos)};
    }
    auto *client = MmcClientDefault::GetInstance();
    if (client == nullptr) {
        for (auto index : validIndices) {
            results[index] = MMC_CLIENT_NOT_INIT;
        }
        return {std::move(results), std::move(infos)};
    }
    std::vector<int> leaseResults;
    std::vector<mmc_data_info> queryInfos;
    const auto ret = client->BatchAddLeaseWithInfo(validKeys, leaseTtlMs, leaseResults, queryInfos);
    if (leaseResults.size() != validKeys.size() || queryInfos.size() != validKeys.size()) {
        // A response may already own server leases. Never retry Add here.
        throw std::runtime_error("unaligned combined read-lease response");
    }
    for (size_t i = 0; i < validIndices.size(); ++i) {
        const auto index = validIndices[i];
        results[index] = ret != MMC_OK && leaseResults[i] == MMC_OK ? ret : leaseResults[i];
        const auto &info = queryInfos[i];
        if (results[index] != MMC_OK || !info.valid) {
            continue;
        }
        KeyInfo keyInfo{info.size, info.numBlobs};
        for (int j = 0; j < info.numBlobs && j < MAX_BLOB_COPIES; ++j) {
            keyInfo.AddLoc(info.ranks[j]);
            keyInfo.AddType(info.types[j]);
            keyInfo.AddGva(info.gvas[j]);
        }
        infos[index] = std::move(keyInfo);
    }
    return {std::move(results), std::move(infos)};
}

int MmcacheStore::BatchRemoveLease(const std::vector<std::string> &keys)
{
    if (keys.empty()) {
        MMC_LOG_ERROR("Invalid batch remove lease input, key size:" << keys.size());
        return MMC_INVALID_PARAM;
    }
    std::vector<const char *> keyArray(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keyArray[i] = keys[i].c_str();
    }
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_REMOVE_LEASE);
    Result ret = mmcc_batch_remove_lease(keyArray.data(), keys.size());
    TP_TRACE_END(TP_MMC_PY_BATCH_REMOVE_LEASE, ret);
    return ret;
}

std::vector<uintptr_t> MmcacheStore::BatchMalloc(const std::vector<std::string> &keys, const std::vector<size_t> &sizes,
                                                 uint16_t media, uint64_t leaseTtlMs)
{
    std::vector<uintptr_t> gvas(keys.size(), 0);
    if (keys.size() != sizes.size()) {
        MMC_LOG_ERROR("Input vector sizes mismatch: keys=" << keys.size() << ", sizes=" << sizes.size());
        return gvas;
    }

    for (const std::string &key : keys) {
        if (key.length() == 0 || key.length() > MAX_KEY_LEN) {
            MMC_LOG_ERROR("Invalid param, key's len (" << key.length() << ") is not between 1 and " << MAX_KEY_LEN);
            return gvas;
        }
    }

    mmc_put_options options{};
    options.mediaType = media;
    options.policy = NATIVE_AFFINITY;
    options.replicaNum = 1;
    std::fill(std::begin(options.preferredLocalServiceIDs), std::end(options.preferredLocalServiceIDs), -1);
    std::vector<const char *> keyArray(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keyArray[i] = keys[i].c_str();
    }
    std::vector<uint64_t> gvaValues(keys.size(), 0);
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_ALLOC);
    Result ret = mmcc_batch_malloc(keyArray.data(), keys.size(), sizes.data(), options, leaseTtlMs, gvaValues.data());
    TP_TRACE_END(TP_MMC_PY_BATCH_ALLOC, ret);
    for (size_t i = 0; i < gvaValues.size(); ++i) {
        gvas[i] = static_cast<uintptr_t>(gvaValues[i]);
    }
    return gvas;
}

int MmcacheStore::BatchCopy(std::vector<void *> &gvas, std::vector<void *> &buffers, std::vector<size_t> &sizes,
                            const int32_t direct)
{
    if (gvas.size() != buffers.size() || gvas.size() != sizes.size()) {
        MMC_LOG_ERROR("Input vector sizes mismatch: gvas=" << gvas.size() << ", buffers=" << buffers.size()
                                                           << ", sizes=" << sizes.size());
        return MMC_INVALID_PARAM;
    }
    std::vector<uint64_t> gvaValues(gvas.size(), 0);
    for (size_t i = 0; i < gvas.size(); ++i) {
        gvaValues[i] = reinterpret_cast<uint64_t>(gvas[i]);
    }
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_COPY);
    Result ret = mmcc_batch_copy(gvaValues.data(), buffers.data(), sizes.data(), sizes.size(), direct);
    TP_TRACE_END(TP_MMC_PY_BATCH_COPY, ret);
    return ret;
}

std::vector<int> MmcacheStore::BatchWriteFinish(const std::vector<std::string> &keys,
                                                const std::vector<int32_t> &writeResults)
{
    if (keys.empty()) {
        MMC_LOG_ERROR("Invalid batch write finish input, key size:" << keys.size());
        return {};
    }
    if (keys.size() != writeResults.size()) {
        MMC_LOG_ERROR("Input vector sizes mismatch: keys=" << keys.size() << ", writeResults=" << writeResults.size());
        return std::vector<int>(keys.size(), MMC_INVALID_PARAM);
    }
    MMC_LOG_DEBUG("BatchWriteFinish enter, keys=" << keys.size());
    std::vector<int> outResults(keys.size(), MMC_INVALID_PARAM);
    std::vector<const char *> keyArray(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        keyArray[i] = keys[i].c_str();
    }
    TP_TRACE_BEGIN(TP_MMC_PY_BATCH_WRITE_FINISH);
    Result ret = mmcc_batch_write_finish(keyArray.data(), keys.size(), writeResults.data(), outResults.data());
    TP_TRACE_END(TP_MMC_PY_BATCH_WRITE_FINISH, ret);
    (void)ret;
    return outResults;
}

} // namespace mmc
} // namespace ock
