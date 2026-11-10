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

#pragma once

#include <map>
#include <utility>
#include <vector>
#include <cctype>

#include "mmc_lock.h"
#include "mmc_ref.h"
#include "mmc_config_validator.h"
#include "mmc_config_convertor.h"
#include "mmc_config_const.h"
#include "mmc_def.h"
#include "mmc_logger.h"
#include "mmc_types.h"
#include "mmc_last_error.h"
#include "smem_bm_def.h"

#ifndef SMEM_BM_FLAG_DRAM_BEST_EFFORT
#define SMEM_BM_FLAG_DRAM_BEST_EFFORT (1U << 10)
#endif

#include "mmc.h"
#include "common/mmc_functions.h"
#include "common/mmc_ip_validator.h"

namespace ock {
namespace mmc {
constexpr uint32_t CONF_MUST = 1;
// Existing MF1.3 HyBM flag; public request/config struct layouts are unchanged.
constexpr uint32_t MMC_BM_FLAG_DRAM_UNRESTRICTED = (1U << 11);
constexpr uint64_t DRAM_SIZE_ALIGNMENT = 2097152; // 2MB
constexpr uint64_t HBM_SIZE_ALIGNMENT = 2097152;  // 2MB

const std::string BOOL_ENUM_STR = "false||true";
const std::string LOG_LEVEL_ENUM_STR = "debug||info||warn||error";
const std::string LOG_OUTPUT_TARGET_ENUM_STR = "screen||file||both";
constexpr int32_t LOG_OUTPUT_TARGET_SCREEN = 0;
constexpr int32_t LOG_OUTPUT_TARGET_FILE = 1;
constexpr int32_t LOG_OUTPUT_TARGET_BOTH = 2;
const std::string LOCAL_SERVER_PROTOCAL_ENUM_STR =
    "host_rdma||host_urma||host_tcp||device_rdma||device_urma||device_uboe||device_sdma||host_shm";

// 定义单位与字节的转换关系
enum class MemUnit { B, KB, MB, GB, TB, UNKNOWN };

enum class ConfValueType {
    VINT = 0,
    VFLOAT = 1,
    VSTRING = 2,
    VBOOL = 3,
    VUINT64 = 4,
};

void StringToUpper(std::string &str);

class Configuration;
using ConfigurationPtr = MmcRef<Configuration>;

class Configuration : public MmcReferable {
public:
    Configuration() = default;
    ~Configuration() override;

    // forbid copy operation
    Configuration(const Configuration &) = delete;
    Configuration &operator=(const Configuration &) = delete;

    // forbid move operation
    Configuration(const Configuration &&) = delete;
    Configuration &operator=(const Configuration &&) = delete;

    bool Setup(const local_config *config);
    bool LoadFromFile(const std::string &filePath);

    static uint64_t ParseMemSize(const std::string &memStr);
    static MemUnit ParseMemUnit(const std::string &unit);

    int32_t GetInt(const std::pair<const char *, int32_t> &item);
    float GetFloat(const std::pair<const char *, float> &item);
    std::string GetString(const std::pair<const char *, const char *> &item);
    bool GetBool(const std::pair<const char *, bool> &item);
    uint64_t GetUInt64(const std::pair<const char *, uint64_t> &item);
    uint64_t GetUInt64(const char *key, uint64_t defaultValue);

    void Set(const std::string &key, int32_t value);
    void Set(const std::string &key, float value);
    void Set(const std::string &key, const std::string &value);
    void Set(const std::string &key, bool value);
    void Set(const std::string &key, uint64_t value);

    bool SetWithTypeAutoConvert(const std::string &key, const std::string &value);

    template<typename T>
    bool SetWithTypeAutoConvert(const std::string &key, const T &value);

    void AddIntConf(const std::pair<std::string, int> &pair, const ValidatorPtr &validator = nullptr,
                    uint32_t flag = CONF_MUST);
    void AddStrConf(const std::pair<std::string, std::string> &pair, const ValidatorPtr &validator = nullptr,
                    uint32_t flag = CONF_MUST);
    void AddBoolConf(const std::pair<std::string, bool> &pair, const ValidatorPtr &validator = nullptr,
                     uint32_t flag = CONF_MUST);
    void AddUInt64Conf(const std::pair<std::string, uint64_t> &pair, const ValidatorPtr &validator = nullptr,
                       uint32_t flag = CONF_MUST);
    void AddConverter(const std::string &key, const ConverterPtr &converter);
    void AddPathConf(const std::pair<std::string, std::string> &pair, const ValidatorPtr &validator = nullptr,
                     uint32_t flag = CONF_MUST);
    std::vector<std::string> ValidateConf();
    void GetAccTlsConfig(mmc_tls_config &tlsConfig);
    void GetHcomTlsConfig(mmc_tls_config &tlsConfig);
    void GetConfigStoreTlsConfig(mmc_tls_config &tlsConfig);
    void GetMetricsTlsConfig(mmc_tls_config &tlsConfig);

    static int ValidateTLSConfig(const mmc_tls_config &tlsConfig);

    const std::string GetBinDir();
    const std::string GetLogPath(const std::string &logPath);
    static int ValidateLogPathConfig(const std::string &logPath);

    bool Initialized() const
    {
        return mInitialized;
    }
    const std::map<std::string, std::string> &GetAllItems() const
    {
        return mAllItems;
    }
    static std::string ResolveUrlField(const std::string &field, const std::string &fieldName = "");

private:
    bool SetWithStrAutoConvert(const std::string &key, const std::string &value);

    void SetValidator(const std::string &key, const ValidatorPtr &validator, uint32_t flag);

    template<class T>
    static void AddValidateError(const ValidatorPtr &validator, std::vector<std::string> &errors, const T &iter)
    {
        if (validator == nullptr) {
            errors.push_back("Failed to validate <" + iter->first + ">, validator is NULL");
            return;
        }
        if (!(validator->Validate(iter->second))) {
            errors.push_back(validator->ErrorMessage());
        }
    }
    void ValidateOneType(const std::string &key, const ValidatorPtr &validator, std::vector<std::string> &errors,
                         ConfValueType &vType);

    void ValidateItem(const std::string &itemKey, std::vector<std::string> &errors);

    void LoadConfigurations();

    void ResolveAllUrlDomains();

    virtual void LoadDefault() {}

    std::string mConfigPath;

    std::map<std::string, int32_t> mIntItems;
    std::map<std::string, float> mFloatItems;
    std::map<std::string, std::string> mStrItems;
    std::map<std::string, bool> mBoolItems;
    std::map<std::string, uint64_t> mUInt64Items;
    std::map<std::string, std::string> mAllItems;

    std::map<std::string, ConfValueType> mValueTypes;
    std::map<std::string, ValidatorPtr> mValueValidator;
    std::map<std::string, ConverterPtr> mValueConverter;

    std::vector<std::pair<std::string, std::string>> mServiceList;
    std::vector<std::string> mMustKeys;
    std::vector<std::string> mLoadDefaultErrors;

    std::vector<std::string> mPathConfs;
    std::vector<std::string> mExceptPrintConfs;
    std::vector<std::string> mInvalidSetConfs;

    bool mInitialized = false;
    Lock mLock;
};

class MetaServiceConfig final : public Configuration {
public:
    void LoadDefault() override
    {
        using namespace ConfConstant;
        AddStrConf(OCK_MMC_META_SERVICE_URL, VNoCheck::Create(), 0);
        AddStrConf(OCK_MMC_META_SERVICE_CONFIG_STORE_URL, VNoCheck::Create(), 0);
        AddStrConf(OCK_MMC_META_SERVICE_HTTP_URL, VNoCheck::Create(), 0);
        AddUInt64Conf(OCK_MMC_METRICS_REPORT_INTERVAL_SECONDS,
                      VUInt64Range::Create(OCK_MMC_METRICS_REPORT_INTERVAL_SECONDS.first, MIN_INTERVAL_SECONDS,
                                           MAX_INTERVAL_SECONDS),
                      0);
        AddUInt64Conf(OCK_MMC_EVICT_CHECK_INTERVAL_SECONDS,
                      VUInt64Range::Create(OCK_MMC_EVICT_CHECK_INTERVAL_SECONDS.first, MIN_INTERVAL_SECONDS,
                                           MAX_INTERVAL_SECONDS),
                      0);
        AddBoolConf(OCK_MMC_META_HA_ENABLE, VStrEnum::Create(OCK_MMC_META_HA_ENABLE.first, BOOL_ENUM_STR), 0);
        AddStrConf(OCK_MMC_LOG_LEVEL, VStrEnum::Create(OCK_MMC_LOG_LEVEL.first, LOG_LEVEL_ENUM_STR), 0);
        AddStrConf(OCK_MMC_LOG_PATH, VStrLength::Create(OCK_MMC_LOG_PATH.first, PATH_MAX_LEN), 0);
        AddIntConf(OCK_MMC_LOG_ROTATION_FILE_SIZE,
                   VIntRange::Create(OCK_MMC_LOG_ROTATION_FILE_SIZE.first, MIN_LOG_ROTATION_FILE_SIZE,
                                     MAX_LOG_ROTATION_FILE_SIZE),
                   0);
        AddIntConf(OCK_MMC_LOG_ROTATION_FILE_COUNT,
                   VIntRange::Create(OCK_MMC_LOG_ROTATION_FILE_COUNT.first, MIN_LOG_ROTATION_FILE_COUNT,
                                     MAX_LOG_ROTATION_FILE_COUNT),
                   0);
        AddStrConf(OCK_MMC_LOG_OUTPUT_TARGET,
                   VStrEnum::Create(OCK_MMC_LOG_OUTPUT_TARGET.first, LOG_OUTPUT_TARGET_ENUM_STR), 0);
        AddIntConf(OKC_MMC_EVICT_THRESHOLD_HIGH,
                   VIntRange::Create(OKC_MMC_EVICT_THRESHOLD_HIGH.first, MIN_EVICT_THRESHOLD, MAX_EVICT_THRESHOLD), 0);
        AddIntConf(OKC_MMC_EVICT_THRESHOLD_LOW,
                   VIntRange::Create(OKC_MMC_EVICT_THRESHOLD_LOW.first, MIN_EVICT_THRESHOLD, MAX_EVICT_THRESHOLD - 1),
                   0);
        AddBoolConf(OCK_MMC_PREFETCH_ENABLED, VStrEnum::Create(OCK_MMC_PREFETCH_ENABLED.first, BOOL_ENUM_STR), 0);
        AddIntConf(
            OCK_MMC_ASYNC_FLUSH_INTERVAL_MS,
            VIntRange::Create(OCK_MMC_ASYNC_FLUSH_INTERVAL_MS.first, MIN_FLUSH_INTERVAL_MS, MAX_FLUSH_INTERVAL_MS), 0);
        AddIntConf(
            OCK_MMC_ASYNC_FLUSH_BATCH_LIMIT,
            VIntRange::Create(OCK_MMC_ASYNC_FLUSH_BATCH_LIMIT.first, MIN_FLUSH_BATCH_LIMIT, MAX_FLUSH_BATCH_LIMIT), 0);
        AddIntConf(OCK_MMC_PENDING_WAIT_TIMEOUT_MS,
                   VIntRange::Create(OCK_MMC_PENDING_WAIT_TIMEOUT_MS.first, MIN_PENDING_WAIT_TIMEOUT_MS,
                                     MAX_PENDING_WAIT_TIMEOUT_MS),
                   0);
        AddIntConf(OCK_MMC_META_LEASE_TTL_MS,
                   VIntRange::Create(OCK_MMC_META_LEASE_TTL_MS.first, MIN_LEASE_TTL_MS, MAX_LEASE_TTL_MS), 0);
        AddIntConf(
            OKC_MMC_EVICT_YIELD_INTERVAL,
            VIntRange::Create(OKC_MMC_EVICT_YIELD_INTERVAL.first, MIN_EVICT_YIELD_INTERVAL, MAX_EVICT_YIELD_INTERVAL),
            0);
        AddIntConf(OKC_MMC_EVICT_YIELD_DURATION_US,
                   VIntRange::Create(OKC_MMC_EVICT_YIELD_DURATION_US.first, MIN_EVICT_YIELD_DURATION_US,
                                     MAX_EVICT_YIELD_DURATION_US),
                   0);

        AddBoolConf(OCK_MMC_DYNAMIC_CONFIG_ENABLE, VStrEnum::Create(OCK_MMC_DYNAMIC_CONFIG_ENABLE.first, BOOL_ENUM_STR),
                    0);
        AddIntConf(OCK_MMC_DYNAMIC_CONFIG_INTERVAL,
                   VIntRange::Create(OCK_MMC_DYNAMIC_CONFIG_INTERVAL.first, MIN_DYNAMIC_CONFIG_INTERVAL,
                                     MAX_DYNAMIC_CONFIG_INTERVAL),
                   0);

        AddBoolConf(OCK_MMC_TLS_ENABLE, VStrEnum::Create(OCK_MMC_TLS_ENABLE.first, BOOL_ENUM_STR), 0);
        AddStrConf(OCK_MMC_TLS_CA_PATH, VStrLength::Create(OCK_MMC_TLS_CA_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_CRL_PATH, VStrLength::Create(OCK_MMC_TLS_CRL_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_CERT_PATH, VStrLength::Create(OCK_MMC_TLS_CERT_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_KEY_PATH, VStrLength::Create(OCK_MMC_TLS_KEY_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_KEY_PASS_PATH, VStrLength::Create(OCK_MMC_TLS_KEY_PASS_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_PACKAGE_PATH, VStrLength::Create(OCK_MMC_TLS_PACKAGE_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_DECRYPTER_PATH, VStrLength::Create(OCK_MMC_TLS_DECRYPTER_PATH.first, TLS_PATH_MAX_LEN),
                   0);

        AddBoolConf(OCK_MMC_CS_TLS_ENABLE, VStrEnum::Create(OCK_MMC_CS_TLS_ENABLE.first, BOOL_ENUM_STR), 0);
        AddStrConf(OCK_MMC_CS_TLS_CA_PATH, VStrLength::Create(OCK_MMC_CS_TLS_CA_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_CRL_PATH, VStrLength::Create(OCK_MMC_CS_TLS_CRL_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_CERT_PATH, VStrLength::Create(OCK_MMC_CS_TLS_CERT_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_KEY_PATH, VStrLength::Create(OCK_MMC_CS_TLS_KEY_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_KEY_PASS_PATH,
                   VStrLength::Create(OCK_MMC_CS_TLS_KEY_PASS_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_PACKAGE_PATH, VStrLength::Create(OCK_MMC_CS_TLS_PACKAGE_PATH.first, TLS_PATH_MAX_LEN),
                   0);
        AddStrConf(OCK_MMC_CS_TLS_DECRYPTER_PATH,
                   VStrLength::Create(OCK_MMC_CS_TLS_DECRYPTER_PATH.first, TLS_PATH_MAX_LEN), 0);

        AddBoolConf(OCK_MMC_KV_EVENTS_ENABLE, VStrEnum::Create(OCK_MMC_KV_EVENTS_ENABLE.first, BOOL_ENUM_STR), 0);
        AddIntConf(OCK_MMC_KV_EVENTS_TRANSPORT,
                   VIntRange::Create(OCK_MMC_KV_EVENTS_TRANSPORT.first, 0, KV_EVENT_TRANSPORT_TYPE_COUNT - 1), 0);
        AddStrConf(OCK_MMC_KV_EVENTS_ENDPOINT, VNoCheck::Create(), 0);
        AddStrConf(OCK_MMC_KV_EVENTS_MODEL_NAME, VNoCheck::Create(), 0);
        AddStrConf(OCK_MMC_KV_EVENTS_TENANT_ID, VNoCheck::Create(), 0);
        AddIntConf(OCK_MMC_KV_EVENTS_BLOCK_SIZE, VNoCheck::Create(), 0);
        AddIntConf(OCK_MMC_KV_EVENTS_QUEUE_CAPACITY, VNoCheck::Create(), 0);
        AddBoolConf(OCK_MMC_KV_EVENTS_HASH_AS_INT, VStrEnum::Create(OCK_MMC_KV_EVENTS_HASH_AS_INT.first, BOOL_ENUM_STR),
                    0);
        AddBoolConf(OCK_MMC_ZMQ_ENABLE_CURVE, VStrEnum::Create(OCK_MMC_ZMQ_ENABLE_CURVE.first, BOOL_ENUM_STR), 0);
        AddStrConf(OCK_MMC_ZMQ_PUBLIC_KEY, VStrLength::Create(OCK_MMC_ZMQ_PUBLIC_KEY.first, CURVE_KEY_Z85_LEN), 0);
        AddStrConf(OCK_MMC_ZMQ_SECRET_KEY, VStrLength::Create(OCK_MMC_ZMQ_SECRET_KEY.first, CURVE_KEY_Z85_LEN), 0);
        AddBoolConf(OCK_MMC_METRICS_TLS_ENABLE, VStrEnum::Create(OCK_MMC_METRICS_TLS_ENABLE.first, BOOL_ENUM_STR), 0);
        AddStrConf(OCK_MMC_METRICS_TLS_CA_PATH, VStrLength::Create(OCK_MMC_METRICS_TLS_CA_PATH.first, TLS_PATH_MAX_LEN),
                   0);
        AddStrConf(OCK_MMC_METRICS_TLS_CRL_PATH,
                   VStrLength::Create(OCK_MMC_METRICS_TLS_CRL_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_METRICS_TLS_CERT_PATH,
                   VStrLength::Create(OCK_MMC_METRICS_TLS_CERT_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_METRICS_TLS_KEY_PATH,
                   VStrLength::Create(OCK_MMC_METRICS_TLS_KEY_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_METRICS_TLS_KEY_PASS_PATH,
                   VStrLength::Create(OCK_MMC_METRICS_TLS_KEY_PASS_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_METRICS_TLS_PACKAGE_PATH,
                   VStrLength::Create(OCK_MMC_METRICS_TLS_PACKAGE_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_METRICS_TLS_DECRYPTER_PATH,
                   VStrLength::Create(OCK_MMC_METRICS_TLS_DECRYPTER_PATH.first, TLS_PATH_MAX_LEN), 0);
    }

    void GetMetaServiceConfig(mmc_meta_service_config_t &config)
    {
        SafeCopy(GetString(ConfConstant::OCK_MMC_META_SERVICE_URL), config.discoveryURL, DISCOVERY_URL_SIZE);
        SafeCopy(GetString(ConfConstant::OCK_MMC_META_SERVICE_CONFIG_STORE_URL), config.configStoreURL,
                 DISCOVERY_URL_SIZE);
        SafeCopy(GetString(ConfConstant::OCK_MMC_META_SERVICE_HTTP_URL), config.httpURL, DISCOVERY_URL_SIZE);

        config.haEnable = GetBool(ConfConstant::OCK_MMC_META_HA_ENABLE);
        std::string logLevelStr = GetString(ConfConstant::OCK_MMC_LOG_LEVEL);
        StringToUpper(logLevelStr);
        config.logLevel = MmcOutLogger::Instance().GetLogLevel(logLevelStr);

        SafeCopy(GetLogPath(GetString(ConfConstant::OCK_MMC_LOG_PATH)), config.logPath, PATH_MAX_SIZE);

        config.evictThresholdHigh = GetInt(ConfConstant::OKC_MMC_EVICT_THRESHOLD_HIGH);
        config.evictThresholdLow = GetInt(ConfConstant::OKC_MMC_EVICT_THRESHOLD_LOW);
        config.prefetchEnabled = GetBool(ConfConstant::OCK_MMC_PREFETCH_ENABLED);
        config.asyncFlushIntervalMs = static_cast<uint32_t>(GetInt(ConfConstant::OCK_MMC_ASYNC_FLUSH_INTERVAL_MS));
        config.asyncFlushBatchLimit = static_cast<uint32_t>(GetInt(ConfConstant::OCK_MMC_ASYNC_FLUSH_BATCH_LIMIT));
        config.pendingWaitTimeoutMs = static_cast<uint64_t>(GetInt(ConfConstant::OCK_MMC_PENDING_WAIT_TIMEOUT_MS));
        config.leaseTtlMs = static_cast<uint64_t>(GetInt(ConfConstant::OCK_MMC_META_LEASE_TTL_MS));
        config.evictYieldInterval = static_cast<uint32_t>(GetInt(ConfConstant::OKC_MMC_EVICT_YIELD_INTERVAL));
        config.evictYieldDurationUs = static_cast<uint32_t>(GetInt(ConfConstant::OKC_MMC_EVICT_YIELD_DURATION_US));
        config.logRotationFileSize = GetInt(ConfConstant::OCK_MMC_LOG_ROTATION_FILE_SIZE) * MB_NUM;
        config.logRotationFileCount = GetInt(ConfConstant::OCK_MMC_LOG_ROTATION_FILE_COUNT);
        std::string logOutputTargetStr = GetString(ConfConstant::OCK_MMC_LOG_OUTPUT_TARGET);
        StringToUpper(logOutputTargetStr);
        if (logOutputTargetStr == "SCREEN") {
            config.logOutputTarget = LOG_OUTPUT_TARGET_SCREEN;
        } else if (logOutputTargetStr == "BOTH") {
            config.logOutputTarget = LOG_OUTPUT_TARGET_BOTH;
        } else {
            config.logOutputTarget = LOG_OUTPUT_TARGET_FILE;
        }
        config.metricsReportIntervalSeconds = GetUInt64(ConfConstant::OCK_MMC_METRICS_REPORT_INTERVAL_SECONDS);
        config.evictCheckIntervalSeconds = GetUInt64(ConfConstant::OCK_MMC_EVICT_CHECK_INTERVAL_SECONDS);
        GetAccTlsConfig(config.accTlsConfig);
        GetConfigStoreTlsConfig(config.configStoreTlsConfig);
        config.kvEvents.enable = GetBool(ConfConstant::OCK_MMC_KV_EVENTS_ENABLE);
        config.kvEvents.transportType = static_cast<int32_t>(GetInt(ConfConstant::OCK_MMC_KV_EVENTS_TRANSPORT));
        SafeCopy(GetString(ConfConstant::OCK_MMC_KV_EVENTS_ENDPOINT), config.kvEvents.endpoint, DISCOVERY_URL_SIZE);
        SafeCopy(GetString(ConfConstant::OCK_MMC_KV_EVENTS_MODEL_NAME), config.kvEvents.modelName, KV_EVENT_FIELD_SIZE);
        SafeCopy(GetString(ConfConstant::OCK_MMC_KV_EVENTS_TENANT_ID), config.kvEvents.tenantId, KV_EVENT_FIELD_SIZE);
        config.kvEvents.blockSize = static_cast<uint32_t>(GetInt(ConfConstant::OCK_MMC_KV_EVENTS_BLOCK_SIZE));
        config.kvEvents.queueCapacity = static_cast<uint32_t>(GetInt(ConfConstant::OCK_MMC_KV_EVENTS_QUEUE_CAPACITY));
        config.kvEvents.hashAsInt = GetBool(ConfConstant::OCK_MMC_KV_EVENTS_HASH_AS_INT);
        config.zmq.enableCurve = GetBool(ConfConstant::OCK_MMC_ZMQ_ENABLE_CURVE);
        SafeCopy(GetString(ConfConstant::OCK_MMC_ZMQ_PUBLIC_KEY), config.zmq.publicKey, KV_EVENT_FIELD_SIZE);
        SafeCopy(GetString(ConfConstant::OCK_MMC_ZMQ_SECRET_KEY), config.zmq.secretKey, KV_EVENT_FIELD_SIZE);
        GetMetricsTlsConfig(config.metricsTlsConfig);
    }
};

class ClientConfig final : public Configuration {
public:
    void LoadDefault() override
    {
        using namespace ConfConstant;
        AddStrConf(OCK_MMC_META_SERVICE_URL, VNoCheck::Create(), 0);
        AddStrConf(OCK_MMC_META_SERVICE_CONFIG_STORE_URL, VNoCheck::Create(), 0);
        AddStrConf(OCK_MMC_LOG_LEVEL, VStrEnum::Create(OCK_MMC_LOG_LEVEL.first, LOG_LEVEL_ENUM_STR), 0);

        AddBoolConf(OCK_MMC_TLS_ENABLE, VStrEnum::Create(OCK_MMC_TLS_ENABLE.first, BOOL_ENUM_STR), 0);
        AddStrConf(OCK_MMC_TLS_CA_PATH, VStrLength::Create(OCK_MMC_TLS_CA_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_CRL_PATH, VStrLength::Create(OCK_MMC_TLS_CRL_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_CERT_PATH, VStrLength::Create(OCK_MMC_TLS_CERT_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_KEY_PATH, VStrLength::Create(OCK_MMC_TLS_KEY_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_KEY_PASS_PATH, VStrLength::Create(OCK_MMC_TLS_KEY_PASS_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_PACKAGE_PATH, VStrLength::Create(OCK_MMC_TLS_PACKAGE_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_TLS_DECRYPTER_PATH, VStrLength::Create(OCK_MMC_TLS_DECRYPTER_PATH.first, TLS_PATH_MAX_LEN),
                   0);

        AddBoolConf(OCK_MMC_CS_TLS_ENABLE, VStrEnum::Create(OCK_MMC_CS_TLS_ENABLE.first, BOOL_ENUM_STR), 0);
        AddStrConf(OCK_MMC_CS_TLS_CA_PATH, VStrLength::Create(OCK_MMC_CS_TLS_CA_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_CRL_PATH, VStrLength::Create(OCK_MMC_CS_TLS_CRL_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_CERT_PATH, VStrLength::Create(OCK_MMC_CS_TLS_CERT_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_KEY_PATH, VStrLength::Create(OCK_MMC_CS_TLS_KEY_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_KEY_PASS_PATH,
                   VStrLength::Create(OCK_MMC_CS_TLS_KEY_PASS_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_CS_TLS_PACKAGE_PATH, VStrLength::Create(OCK_MMC_CS_TLS_PACKAGE_PATH.first, TLS_PATH_MAX_LEN),
                   0);
        AddStrConf(OCK_MMC_CS_TLS_DECRYPTER_PATH,
                   VStrLength::Create(OCK_MMC_CS_TLS_DECRYPTER_PATH.first, TLS_PATH_MAX_LEN), 0);

        AddIntConf(OKC_MMC_LOCAL_SERVICE_WORLD_SIZE,
                   VIntRange::Create(OKC_MMC_LOCAL_SERVICE_WORLD_SIZE.first, MIN_WORLD_SIZE, MAX_WORLD_SIZE), 0);
        AddStrConf(OKC_MMC_LOCAL_SERVICE_BM_IP_PORT, VNoCheck::Create(), 0);
        AddStrConf(OKC_MMC_LOCAL_SERVICE_BM_IP_PORT_DOMAIN, VNoCheck::Create(), 0);

        AddStrConf(OKC_MMC_LOCAL_SERVICE_PROTOCOL,
                   VStrEnum::Create(OKC_MMC_LOCAL_SERVICE_PROTOCOL.first, LOCAL_SERVER_PROTOCAL_ENUM_STR),
                   1); // REQUIRED
        AddStrConf(OKC_MMC_LOCAL_SERVICE_TAG, VStrLength::Create(OKC_MMC_LOCAL_SERVICE_TAG.first, MMC_TAG_SIZE - 1), 0);
        AddStrConf(OKC_MMC_LOCAL_SERVICE_TAG_OP_INFO,
                   VStrLength::Create(OKC_MMC_LOCAL_SERVICE_TAG_OP_INFO.first, MMC_TAG_OP_INFO_SIZE - 1), 0);
        AddStrConf(OKC_MMC_LOCAL_SERVICE_DRAM_SIZE, VNoCheck::Create(), 1); // REQUIRED
        AddStrConf(OKC_MMC_LOCAL_SERVICE_MAX_DRAM_SIZE, VNoCheck::Create(), 0);
        AddStrConf(OKC_MMC_LOCAL_SERVICE_HBM_SIZE, VNoCheck::Create(), 0);
        AddStrConf(OKC_MMC_LOCAL_SERVICE_MAX_HBM_SIZE, VNoCheck::Create(), 0);

        // HCOM TLS config
        AddStrConf(OKC_MMC_LOCAL_SERVICE_BM_HCOM_URL, VNoCheck::Create(), 0);
        AddStrConf(OKC_MMC_LOCAL_SERVICE_BACKEND_ID, VNoCheck::Create(), 0);
        AddBoolConf(OCK_MMC_HCOM_TLS_ENABLE, VStrEnum::Create(OCK_MMC_HCOM_TLS_ENABLE.first, BOOL_ENUM_STR), 0);
        AddStrConf(OCK_MMC_HCOM_TLS_CA_PATH, VStrLength::Create(OCK_MMC_HCOM_TLS_CA_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_HCOM_TLS_CRL_PATH, VStrLength::Create(OCK_MMC_HCOM_TLS_CRL_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_HCOM_TLS_CERT_PATH, VStrLength::Create(OCK_MMC_HCOM_TLS_CERT_PATH.first, TLS_PATH_MAX_LEN),
                   0);
        AddStrConf(OCK_MMC_HCOM_TLS_KEY_PATH, VStrLength::Create(OCK_MMC_HCOM_TLS_KEY_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_HCOM_TLS_KEY_PASS_PATH,
                   VStrLength::Create(OCK_MMC_HCOM_TLS_KEY_PASS_PATH.first, TLS_PATH_MAX_LEN), 0);
        AddStrConf(OCK_MMC_HCOM_TLS_DECRYPTER_PATH,
                   VStrLength::Create(OCK_MMC_HCOM_TLS_DECRYPTER_PATH.first, TLS_PATH_MAX_LEN), 0);

        AddIntConf(OKC_MMC_CLIENT_RETRY_MILLISECONDS,
                   VIntRange::Create(OKC_MMC_CLIENT_RETRY_MILLISECONDS.first, MIN_RETRY_MS, MAX_RETRY_MS), 0);
        AddIntConf(OCK_MMC_CLIENT_TIMEOUT_SECONDS,
                   VIntRange::Create(OCK_MMC_CLIENT_TIMEOUT_SECONDS.first, MIN_TIMEOUT_SEC, MAX_TIMEOUT_SEC), 0);
        AddIntConf(
            OCK_MMC_CLIENT_READ_THREAD_POOL_SIZE,
            VIntRange::Create(OCK_MMC_CLIENT_READ_THREAD_POOL_SIZE.first, MIN_THREAD_POOL_SIZE, MAX_THREAD_POOL_SIZE),
            0);
        AddBoolConf(OCK_MMC_CLIENT_AGGREGATE_IO, VStrEnum::Create(OCK_MMC_CLIENT_AGGREGATE_IO.first, BOOL_ENUM_STR), 0);
        AddIntConf(
            OCK_MMC_CLIENT_WRITE_THREAD_POOL_SIZE,
            VIntRange::Create(OCK_MMC_CLIENT_WRITE_THREAD_POOL_SIZE.first, MIN_THREAD_POOL_SIZE, MAX_THREAD_POOL_SIZE),
            0);
        AddBoolConf(OCK_MMC_LOCAL_SERVICE_STORAGE_ENABLED,
                    VStrEnum::Create(OCK_MMC_LOCAL_SERVICE_STORAGE_ENABLED.first, BOOL_ENUM_STR), 0);
        AddBoolConf(OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED,
                    VStrEnum::Create(OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED.first, BOOL_ENUM_STR), 0);
        AddBoolConf(OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT,
                    VStrEnum::Create(OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT.first, BOOL_ENUM_STR), 0);
        AddIntConf(OCK_MMC_CLIENT_AGGREGATE_NUM,
                   VIntRange::Create(OCK_MMC_CLIENT_AGGREGATE_NUM.first, 1, MAX_AGGREGATE_NUM), 0);
        AddStrConf(OCK_MMC_CLIENT_BATCH_CHUNK_SIZE, VNoCheck::Create(), 0);
        AddIntConf(OCK_MMC_CLIENT_BATCH_CHUNK_COUNT,
                   VIntRange::Create(OCK_MMC_CLIENT_BATCH_CHUNK_COUNT.first, 1, MAX_BATCH_CHUNK_COUNT), 0);

        AddBoolConf(OCK_MMC_DYNAMIC_CONFIG_ENABLE, VStrEnum::Create(OCK_MMC_DYNAMIC_CONFIG_ENABLE.first, BOOL_ENUM_STR),
                    0);
        AddIntConf(OCK_MMC_DYNAMIC_CONFIG_INTERVAL,
                   VIntRange::Create(OCK_MMC_DYNAMIC_CONFIG_INTERVAL.first, MIN_DYNAMIC_CONFIG_INTERVAL,
                                     MAX_DYNAMIC_CONFIG_INTERVAL),
                   0);
    }

    void GetLocalServiceConfig(mmc_local_service_config_t &config)
    {
        SafeCopy(GetString(ConfConstant::OCK_MMC_META_SERVICE_URL), config.discoveryURL, DISCOVERY_URL_SIZE);

        config.worldSize = static_cast<uint32_t>(GetInt(ConfConstant::OKC_MMC_LOCAL_SERVICE_WORLD_SIZE));
        SafeCopy(GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_BM_IP_PORT), config.bmIpPort, DISCOVERY_URL_SIZE);
        SafeCopy(GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_BM_IP_PORT_DOMAIN), config.bmIpPortDomain,
                 DISCOVERY_URL_SIZE);
        SafeCopy(GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_BM_HCOM_URL), config.bmHcomUrl, DISCOVERY_URL_SIZE);
        SafeCopy(GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_BACKEND_ID), config.backendId, DISCOVERY_URL_SIZE);
        config.createId = 0;
        SafeCopy(GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_PROTOCOL), config.dataOpType, PROTOCOL_SIZE);
        SafeCopy(GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_TAG), config.tag, MMC_TAG_SIZE);
        SafeCopy(GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_TAG_OP_INFO), config.tagOpInfo, MMC_TAG_OP_INFO_SIZE);
        config.localDRAMSize = GetUInt64(ConfConstant::OKC_MMC_LOCAL_SERVICE_DRAM_SIZE.first, GB_MEM_BYTES);
        config.localMaxDRAMSize =
            GetUInt64(ConfConstant::OKC_MMC_LOCAL_SERVICE_MAX_DRAM_SIZE.first, config.localDRAMSize);
        config.localHBMSize = GetUInt64(ConfConstant::OKC_MMC_LOCAL_SERVICE_HBM_SIZE.first, 0);
        config.localMaxHBMSize = GetUInt64(ConfConstant::OKC_MMC_LOCAL_SERVICE_MAX_HBM_SIZE.first, config.localHBMSize);
        config.storageEnabled = GetBool(ConfConstant::OCK_MMC_LOCAL_SERVICE_STORAGE_ENABLED);
        if (GetBool(ConfConstant::OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED)) {
            config.flags |= MMC_BM_FLAG_DRAM_UNRESTRICTED;
        }
        if (GetBool(ConfConstant::OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT)) {
            config.flags |= SMEM_BM_FLAG_DRAM_BEST_EFFORT;
        }
        auto protocol = std::string(config.dataOpType);
        std::string logLevelStr = GetString(ConfConstant::OCK_MMC_LOG_LEVEL);
        StringToUpper(logLevelStr);
        config.logLevel = MmcOutLogger::Instance().GetLogLevel(logLevelStr);
        GetAccTlsConfig(config.accTlsConfig);
        GetHcomTlsConfig(config.hcomTlsConfig);
        GetConfigStoreTlsConfig(config.configStoreTlsConfig);
        config.dynamicConfigEnable = GetBool(ConfConstant::OCK_MMC_DYNAMIC_CONFIG_ENABLE);
        config.dynamicConfigInterval = static_cast<uint32_t>(GetInt(ConfConstant::OCK_MMC_DYNAMIC_CONFIG_INTERVAL));
    }

    void GetClientConfig(mmc_client_config_t &config)
    {
        SafeCopy(GetString(ConfConstant::OCK_MMC_META_SERVICE_URL), config.discoveryURL, DISCOVERY_URL_SIZE);

        config.rpcRetryTimeOut = static_cast<uint32_t>(GetInt(ConfConstant::OKC_MMC_CLIENT_RETRY_MILLISECONDS));
        config.timeOut = static_cast<uint32_t>(GetInt(ConfConstant::OCK_MMC_CLIENT_TIMEOUT_SECONDS));
        config.readThreadPoolNum = static_cast<uint32_t>(GetInt(ConfConstant::OCK_MMC_CLIENT_READ_THREAD_POOL_SIZE));
        config.aggregateIO = GetBool(ConfConstant::OCK_MMC_CLIENT_AGGREGATE_IO);
        config.aggregateNum = GetInt(ConfConstant::OCK_MMC_CLIENT_AGGREGATE_NUM);
        config.writeThreadPoolNum = static_cast<uint32_t>(GetInt(ConfConstant::OCK_MMC_CLIENT_WRITE_THREAD_POOL_SIZE));
        config.batchChunkSize =
            GetUInt64(ConfConstant::OCK_MMC_CLIENT_BATCH_CHUNK_SIZE.first, DEFAULT_BATCH_CHUNK_SIZE);
        config.batchChunkCount = GetInt(ConfConstant::OCK_MMC_CLIENT_BATCH_CHUNK_COUNT);
        std::string logLevelStr = GetString(ConfConstant::OCK_MMC_LOG_LEVEL);
        StringToUpper(logLevelStr);
        config.logLevel = MmcOutLogger::Instance().GetLogLevel(logLevelStr);
        GetAccTlsConfig(config.tlsConfig);
        SafeCopy(GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_PROTOCOL), config.dataOpType, PROTOCOL_SIZE);
    }

    static Result ValidateLocalServiceConfig(mmc_local_service_config_t &config)
    {
        constexpr uint64_t GB_SIZE_ALIGNMENT = 1073741824; // 1GB
        uint64_t alignment = DRAM_SIZE_ALIGNMENT;          // 默认 2MB 对齐
        std::string protocol(config.dataOpType);
        if ((config.flags & MMC_BM_FLAG_DRAM_UNRESTRICTED) != 0U && protocol != "device_sdma") {
            MMC_LOG_ERROR("dram.unrestricted requires device_sdma, protocol:" << protocol);
            return MMC_INVALID_PARAM;
        }

        if (protocol == "device_rdma" || protocol == "device_urma" || protocol == "device_uboe" ||
            protocol == "device_sdma") {
            alignment = GB_SIZE_ALIGNMENT;
        }

        MMC_LOG_DEBUG("Before alignment " << (alignment == GB_SIZE_ALIGNMENT ? "1GB" : "2MB")
                                          << ", ock.mmc.local_service.dram.size is " << config.localDRAMSize
                                          << ", ock.mmc.local_service.max.dram.size is " << config.localMaxDRAMSize
                                          << ", ock.mmc.local_service.hbm.size is " << config.localHBMSize
                                          << ", ock.mmc.local_service.max.hbm.size is " << config.localMaxHBMSize);

        auto align_up = [](uint64_t size, uint64_t align) {
            if (size == 0)
                return size;
            return (size + align - 1) / align * align;
        };

        config.localDRAMSize = align_up(config.localDRAMSize, alignment);
        config.localMaxDRAMSize = align_up(config.localMaxDRAMSize, alignment);
        config.localHBMSize = align_up(config.localHBMSize, alignment);
        config.localMaxHBMSize = align_up(config.localMaxHBMSize, alignment);

        if (config.localDRAMSize > MAX_DRAM_SIZE) {
            MMC_LOG_ERROR("After alignment " << (alignment == GB_SIZE_ALIGNMENT ? "1GB" : "2MB")
                                             << ", ock.mmc.local_service.dram.size(" << config.localDRAMSize
                                             << ") exceeds (" << MAX_DRAM_SIZE << ")");
            return MMC_INVALID_PARAM;
        }
        if (config.localMaxDRAMSize > MAX_DRAM_SIZE) {
            MMC_LOG_ERROR("After alignment " << (alignment == GB_SIZE_ALIGNMENT ? "1GB" : "2MB")
                                             << ", ock.mmc.local_service.max.dram.size(" << config.localMaxDRAMSize
                                             << ") exceeds (" << MAX_DRAM_SIZE << ")");
            return MMC_INVALID_PARAM;
        }
        if (config.localMaxDRAMSize < config.localDRAMSize) {
            MMC_LOG_ERROR("ock.mmc.local_service.max.dram.size(" << config.localMaxDRAMSize
                                                                 << ") is smaller than ock.mmc.local_service.dram.size("
                                                                 << config.localDRAMSize << ")");
            return MMC_INVALID_PARAM;
        }

        if (config.localHBMSize > MAX_HBM_SIZE) {
            MMC_LOG_ERROR("After alignment " << (alignment == GB_SIZE_ALIGNMENT ? "1GB" : "2MB")
                                             << ", ock.mmc.local_service.hbm.size(" << config.localHBMSize
                                             << ") exceeds (" << MAX_HBM_SIZE << ")");
            return MMC_INVALID_PARAM;
        }
        if (config.localMaxHBMSize > MAX_HBM_SIZE) {
            MMC_LOG_ERROR("After alignment " << (alignment == GB_SIZE_ALIGNMENT ? "1GB" : "2MB")
                                             << ", ock.mmc.local_service.max.hbm.size(" << config.localMaxHBMSize
                                             << ") exceeds (" << MAX_HBM_SIZE << ")");
            return MMC_INVALID_PARAM;
        }
        if (config.localMaxHBMSize < config.localHBMSize) {
            MMC_LOG_ERROR("ock.mmc.local_service.max.hbm.size(" << config.localMaxHBMSize
                                                                << ") is smaller than ock.mmc.local_service.hbm.size("
                                                                << config.localHBMSize << ");");
            return MMC_INVALID_PARAM;
        }

        MMC_LOG_INFO("After alignment " << (alignment == GB_SIZE_ALIGNMENT ? "1GB" : "2MB")
                                        << ", ock.mmc.local_service.dram.size is " << config.localDRAMSize
                                        << ", ock.mmc.local_service.max.dram.size is " << config.localMaxDRAMSize);
        MMC_LOG_INFO("After alignment " << (alignment == GB_SIZE_ALIGNMENT ? "1GB" : "2MB")
                                        << ", ock.mmc.local_service.hbm.size is " << config.localHBMSize
                                        << ", ock.mmc.local_service.max.hbm.size is " << config.localMaxHBMSize);

        MMC_VALIDATE_RETURN(ValidateTLSConfig(config.accTlsConfig) == MMC_OK, "Invalid acc_link TLS config",
                            MMC_INVALID_PARAM);
        MMC_VALIDATE_RETURN(ValidateTLSConfig(config.hcomTlsConfig) == MMC_OK, "Invalid hcom TLS config",
                            MMC_INVALID_PARAM);
        MMC_VALIDATE_RETURN(ValidateTLSConfig(config.configStoreTlsConfig) == MMC_OK, "Invalid config store TLS config",
                            MMC_INVALID_PARAM);

        return MMC_OK;
    }
};

} // namespace mmc
} // namespace ock
