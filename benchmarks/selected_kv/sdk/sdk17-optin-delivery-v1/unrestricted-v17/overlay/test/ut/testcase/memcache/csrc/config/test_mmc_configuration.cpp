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
#include "gtest/gtest.h"

#define private public
#include "mmc_smem_bm_helper.h"
#include "mmc_configuration.h"
#include "mmc_def.h"
#undef private

#include "mmc_meta_metric_manager.h"

using namespace testing;
using namespace std;
using namespace ock::mmc;

class TestMmcConfiguration : public testing::Test {
public:
    TestMmcConfiguration();

    void SetUp() override;

    void TearDown() override;

protected:
};
TestMmcConfiguration::TestMmcConfiguration() = default;

void TestMmcConfiguration::SetUp() {}

void TestMmcConfiguration::TearDown() {}

static local_config CreateLocalConfigWithCurrentDefaults()
{
    local_config config{};
    SafeCopy("tcp://127.0.0.1:5000", config.meta_service_url, sizeof(config.meta_service_url));
    SafeCopy("tcp://127.0.0.1:6000", config.config_store_url, sizeof(config.config_store_url));
    SafeCopy("", config.backend_id, sizeof(config.backend_id));
    SafeCopy("info", config.log_level, sizeof(config.log_level));
    config.world_size = 256UL;
    SafeCopy("host_rdma", config.protocol, sizeof(config.protocol));
    SafeCopy("tcp://127.0.0.1:7000", config.hcom_url, sizeof(config.hcom_url));
    SafeCopy("1GB", config.dram_size, sizeof(config.dram_size));
    SafeCopy("0", config.hbm_size, sizeof(config.hbm_size));
    SafeCopy("1TB", config.max_dram_size, sizeof(config.max_dram_size));
    SafeCopy("0", config.max_hbm_size, sizeof(config.max_hbm_size));
    config.client_retry_milliseconds = 0;
    config.client_timeout_seconds = 60UL;
    config.read_thread_pool_size = 32UL;
    config.write_thread_pool_size = 4UL;
    config.aggregate_io = true;
    config.aggregate_num = 122UL;
    config.tls_enable = false;
    config.config_store_tls_enable = false;
    config.hcom_tls_enable = false;
    config.dynamic_config_enable = true;
    config.dynamic_config_interval = DEFAULT_DYNAMIC_CONFIG_INTERVAL;
    return config;
}

TEST_F(TestMmcConfiguration, ValidateTLSConfigTest)
{
    mmc_tls_config tlsConfig{};
    tlsConfig.tlsEnable = true;
    // 1. caPath is empty
    auto ret = Configuration::ValidateTLSConfig(tlsConfig);
    ASSERT_NE(ret, MMC_OK);

    // 2. certPath is empty
    SafeCopy("./", tlsConfig.caPath, TLS_PATH_SIZE);
    ret = Configuration::ValidateTLSConfig(tlsConfig);
    ASSERT_NE(ret, MMC_OK);

    // 3. keyPath is empty
    SafeCopy("./", tlsConfig.certPath, TLS_PATH_SIZE);
    ret = Configuration::ValidateTLSConfig(tlsConfig);
    ASSERT_NE(ret, MMC_OK);

    // 4. crlPath is error
    SafeCopy("./", tlsConfig.keyPath, TLS_PATH_SIZE);
    SafeCopy("nonPath", tlsConfig.crlPath, TLS_PATH_SIZE);
    ret = Configuration::ValidateTLSConfig(tlsConfig);
    ASSERT_NE(ret, MMC_OK);

    // 4. keyPassPath is error
    SafeCopy("./", tlsConfig.crlPath, TLS_PATH_SIZE);
    SafeCopy("nonPath", tlsConfig.keyPassPath, TLS_PATH_SIZE);
    ret = Configuration::ValidateTLSConfig(tlsConfig);
    ASSERT_NE(ret, MMC_OK);

    // 5. packagePath is error
    SafeCopy("./", tlsConfig.keyPassPath, TLS_PATH_SIZE);
    SafeCopy("nonPath", tlsConfig.packagePath, TLS_PATH_SIZE);
    ret = Configuration::ValidateTLSConfig(tlsConfig);
    ASSERT_NE(ret, MMC_OK);

    // 6. decrypterLibPath is error
    SafeCopy("./", tlsConfig.packagePath, TLS_PATH_SIZE);
    SafeCopy("nonPath", tlsConfig.decrypterLibPath, TLS_PATH_SIZE);
    ret = Configuration::ValidateTLSConfig(tlsConfig);
    ASSERT_NE(ret, MMC_OK);

    SafeCopy("./", tlsConfig.decrypterLibPath, TLS_PATH_SIZE);
    ret = Configuration::ValidateTLSConfig(tlsConfig);
    ASSERT_EQ(ret, MMC_OK);
}

TEST_F(TestMmcConfiguration, GetLogPathTest)
{
    Configuration configuration;

    auto logPath = configuration.GetLogPath("");
    ASSERT_NE(logPath, "");

    logPath = configuration.GetLogPath("/");
    ASSERT_EQ(logPath, "/");

    logPath = configuration.GetLogPath("./");
    ASSERT_NE(logPath, "/");
}

TEST_F(TestMmcConfiguration, EvictYieldConfigDefaultsTest)
{
    MetaServiceConfig config;
    config.LoadDefault();

    EXPECT_EQ(config.GetInt(ConfConstant::OKC_MMC_EVICT_YIELD_INTERVAL), 0);
    EXPECT_EQ(config.GetInt(ConfConstant::OKC_MMC_EVICT_YIELD_DURATION_US), 100);
}

TEST_F(TestMmcConfiguration, EvictYieldConfigValidationTest)
{
    MetaServiceConfig config;
    config.LoadDefault();

    config.Set(ConfConstant::OKC_MMC_EVICT_YIELD_INTERVAL.first, static_cast<int32_t>(100));
    config.Set(ConfConstant::OKC_MMC_EVICT_YIELD_DURATION_US.first, static_cast<int32_t>(200));

    mmc_meta_service_config_t metaConfig{};
    config.GetMetaServiceConfig(metaConfig);

    EXPECT_EQ(metaConfig.evictYieldInterval, 100U);
    EXPECT_EQ(metaConfig.evictYieldDurationUs, 200U);
}

TEST_F(TestMmcConfiguration, EvictYieldConfigRangeValidationTest)
{
    MetaServiceConfig config;
    config.LoadDefault();

    config.Set(ConfConstant::OKC_MMC_EVICT_YIELD_INTERVAL.first, static_cast<int32_t>(-1));
    auto errors = config.ValidateConf();
    ASSERT_FALSE(errors.empty());

    config.Set(ConfConstant::OKC_MMC_EVICT_YIELD_INTERVAL.first, static_cast<int32_t>(100001));
    errors = config.ValidateConf();
    ASSERT_FALSE(errors.empty());

    config.Set(ConfConstant::OKC_MMC_EVICT_YIELD_INTERVAL.first, static_cast<int32_t>(100));
    config.Set(ConfConstant::OKC_MMC_EVICT_YIELD_DURATION_US.first, static_cast<int32_t>(10001));
    errors = config.ValidateConf();
    ASSERT_FALSE(errors.empty());

    config.Set(ConfConstant::OKC_MMC_EVICT_YIELD_DURATION_US.first, static_cast<int32_t>(200));
    errors = config.ValidateConf();
    ASSERT_TRUE(errors.empty());
}

TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigTest)
{
    // 测试设备类型协议（1GB对齐）
    mmc_local_service_config_t deviceConfig{};
    deviceConfig.logLevel = INFO_LEVEL;
    deviceConfig.accTlsConfig.tlsEnable = false;
    deviceConfig.hcomTlsConfig.tlsEnable = false;
    deviceConfig.configStoreTlsConfig.tlsEnable = false;

    // 测试device_rdma协议，1GB向上对齐
    SafeCopy("device_rdma", deviceConfig.dataOpType, PROTOCOL_SIZE);
    deviceConfig.localDRAMSize = 1536 * 1024 * 1024;       // 1.5GB，应该对齐到2GB
    deviceConfig.localMaxDRAMSize = 3584ULL * 1024 * 1024; // 3.5GB，应该对齐到4GB
    deviceConfig.localHBMSize = 2048ULL * 1024 * 1024;     // 2GB，已经对齐
    deviceConfig.localMaxHBMSize = 4096ULL * 1024 * 1024;  // 4GB，已经对齐

    auto ret = ClientConfig::ValidateLocalServiceConfig(deviceConfig);
    ASSERT_EQ(ret, MMC_OK);
    ASSERT_EQ(deviceConfig.localDRAMSize, 2ULL * 1024 * 1024 * 1024);    // 2GB
    ASSERT_EQ(deviceConfig.localMaxDRAMSize, 4ULL * 1024 * 1024 * 1024); // 4GB
    ASSERT_EQ(deviceConfig.localHBMSize, 2ULL * 1024 * 1024 * 1024);     // 2GB
    ASSERT_EQ(deviceConfig.localMaxHBMSize, 4ULL * 1024 * 1024 * 1024);  // 4GB

    // 测试device_urma协议，1GB向上对齐
    SafeCopy("device_urma", deviceConfig.dataOpType, PROTOCOL_SIZE);
    deviceConfig.localDRAMSize = 768ULL * 1024ULL * 1024ULL;     // 768MB，应该对齐到1GB
    deviceConfig.localMaxDRAMSize = 1024ULL * 1024ULL * 1024ULL; // 1GB，已经对齐
    deviceConfig.localHBMSize = 256ULL * 1024ULL * 1024ULL;      // 256MB，应该对齐到1GB
    deviceConfig.localMaxHBMSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;

    ret = ClientConfig::ValidateLocalServiceConfig(deviceConfig);
    ASSERT_EQ(ret, MMC_OK);
    ASSERT_EQ(deviceConfig.localDRAMSize, 1ULL * 1024ULL * 1024ULL * 1024ULL);
    ASSERT_EQ(deviceConfig.localMaxDRAMSize, 1ULL * 1024ULL * 1024ULL * 1024ULL);
    ASSERT_EQ(deviceConfig.localHBMSize, 1ULL * 1024ULL * 1024ULL * 1024ULL);
    ASSERT_EQ(deviceConfig.localMaxHBMSize, 2ULL * 1024ULL * 1024ULL * 1024ULL);
    ASSERT_EQ(MmcSmemBmHelper::TransSmemBmDataOpType("device_urma"), SMEMB_DATA_OP_DEVICE_URMA);

    // 测试device_uboe协议，1GB向上对齐
    SafeCopy("device_uboe", deviceConfig.dataOpType, PROTOCOL_SIZE);
    deviceConfig.localDRAMSize = 768ULL * 1024ULL * 1024ULL;     // 768MB，应该对齐到1GB
    deviceConfig.localMaxDRAMSize = 1024ULL * 1024ULL * 1024ULL; // 1GB，已经对齐
    deviceConfig.localHBMSize = 256ULL * 1024ULL * 1024ULL;      // 256MB，应该对齐到1GB
    deviceConfig.localMaxHBMSize = 2ULL * 1024ULL * 1024ULL * 1024ULL;

    ret = ClientConfig::ValidateLocalServiceConfig(deviceConfig);
    ASSERT_EQ(ret, MMC_OK);
    ASSERT_EQ(deviceConfig.localDRAMSize, 1ULL * 1024ULL * 1024ULL * 1024ULL);
    ASSERT_EQ(deviceConfig.localMaxDRAMSize, 1ULL * 1024ULL * 1024ULL * 1024ULL);
    ASSERT_EQ(deviceConfig.localHBMSize, 1ULL * 1024ULL * 1024ULL * 1024ULL);
    ASSERT_EQ(deviceConfig.localMaxHBMSize, 2ULL * 1024ULL * 1024ULL * 1024ULL);
    ASSERT_EQ(MmcSmemBmHelper::TransSmemBmDataOpType("device_uboe"), SMEMB_DATA_OP_DEVICE_UBOE);

    // 测试device_sdma协议，1GB向上对齐
    SafeCopy("device_sdma", deviceConfig.dataOpType, PROTOCOL_SIZE);
    deviceConfig.localDRAMSize = 512ULL * 1024 * 1024;     // 512MB，应该对齐到1GB
    deviceConfig.localMaxDRAMSize = 1024ULL * 1024 * 1024; // 1GB，已经对齐
    deviceConfig.localHBMSize = 128ULL * 1024 * 1024;      // 128MB，应该对齐到1GB
    deviceConfig.localMaxHBMSize = 2048ULL * 1024 * 1024;  // 2GB，已经对齐

    ret = ClientConfig::ValidateLocalServiceConfig(deviceConfig);
    ASSERT_EQ(ret, MMC_OK);
    ASSERT_EQ(deviceConfig.localDRAMSize, 1ULL * 1024 * 1024 * 1024);    // 1GB
    ASSERT_EQ(deviceConfig.localMaxDRAMSize, 1ULL * 1024 * 1024 * 1024); // 1GB
    ASSERT_EQ(deviceConfig.localHBMSize, 1ULL * 1024 * 1024 * 1024);     // 1GB
    ASSERT_EQ(deviceConfig.localMaxHBMSize, 2ULL * 1024 * 1024 * 1024);  // 2GB

    // 测试主机类型协议（2MB对齐）
    mmc_local_service_config_t hostConfig{};
    hostConfig.logLevel = INFO_LEVEL;
    hostConfig.accTlsConfig.tlsEnable = false;
    hostConfig.hcomTlsConfig.tlsEnable = false;
    hostConfig.configStoreTlsConfig.tlsEnable = false;

    // 测试host_rdma协议，2MB向上对齐
    SafeCopy("host_rdma", hostConfig.dataOpType, PROTOCOL_SIZE);
    hostConfig.localDRAMSize = 3ULL * 1024 * 1024;    // 3MB，应该对齐到4MB
    hostConfig.localMaxDRAMSize = 5ULL * 1024 * 1024; // 5MB，应该对齐到6MB
    hostConfig.localHBMSize = 2ULL * 1024 * 1024;     // 2MB，已经对齐
    hostConfig.localMaxHBMSize = 4ULL * 1024 * 1024;  // 4MB，已经对齐

    ret = ClientConfig::ValidateLocalServiceConfig(hostConfig);
    ASSERT_EQ(ret, MMC_OK);
    ASSERT_EQ(hostConfig.localDRAMSize, 4ULL * 1024 * 1024);    // 4MB
    ASSERT_EQ(hostConfig.localMaxDRAMSize, 6ULL * 1024 * 1024); // 6MB
    ASSERT_EQ(hostConfig.localHBMSize, 2ULL * 1024 * 1024);     // 2MB
    ASSERT_EQ(hostConfig.localMaxHBMSize, 4ULL * 1024 * 1024);  // 4MB

    // 测试host_tcp协议，2MB向上对齐
    SafeCopy("host_tcp", hostConfig.dataOpType, PROTOCOL_SIZE);
    hostConfig.localDRAMSize = 1ULL * 1024 * 1024;    // 1MB，应该对齐到2MB
    hostConfig.localMaxDRAMSize = 3ULL * 1024 * 1024; // 3MB，应该对齐到4MB
    hostConfig.localHBMSize = 512ULL * 1024;          // 512KB，应该对齐到2MB
    hostConfig.localMaxHBMSize = 1536ULL * 1024;      // 1.5MB，应该对齐到2MB

    ret = ClientConfig::ValidateLocalServiceConfig(hostConfig);
    ASSERT_EQ(ret, MMC_OK);
    ASSERT_EQ(hostConfig.localDRAMSize, 2ULL * 1024 * 1024);    // 2MB
    ASSERT_EQ(hostConfig.localMaxDRAMSize, 4ULL * 1024 * 1024); // 4MB
    ASSERT_EQ(hostConfig.localHBMSize, 2ULL * 1024 * 1024);     // 2MB
    ASSERT_EQ(hostConfig.localMaxHBMSize, 2ULL * 1024 * 1024);  // 2MB

    // 测试边界情况：配置为0
    mmc_local_service_config_t zeroConfig{};
    zeroConfig.logLevel = INFO_LEVEL;
    zeroConfig.accTlsConfig.tlsEnable = false;
    zeroConfig.hcomTlsConfig.tlsEnable = false;
    zeroConfig.configStoreTlsConfig.tlsEnable = false;
    SafeCopy("host_rdma", zeroConfig.dataOpType, PROTOCOL_SIZE);
    zeroConfig.localDRAMSize = 0;
    zeroConfig.localMaxDRAMSize = 0;
    zeroConfig.localHBMSize = 0;
    zeroConfig.localMaxHBMSize = 0;

    ret = ClientConfig::ValidateLocalServiceConfig(zeroConfig);
    ASSERT_EQ(ret, MMC_OK); // 两者都为0是允许的（分离部署场景）

    // 测试一个为0，一个不为0的情况
    zeroConfig.localDRAMSize = 2ULL * 1024 * 1024;    // 2MB
    zeroConfig.localMaxDRAMSize = 2ULL * 1024 * 1024; // 2MB
    zeroConfig.localHBMSize = 0;
    zeroConfig.localMaxHBMSize = 0;

    ret = ClientConfig::ValidateLocalServiceConfig(zeroConfig);
    ASSERT_EQ(ret, MMC_OK);
    ASSERT_EQ(zeroConfig.localDRAMSize, 2ULL * 1024 * 1024); // 2MB
    ASSERT_EQ(zeroConfig.localHBMSize, 0);                   // 保持为0
}

TEST_F(TestMmcConfiguration, SetTest)
{
    Configuration configuration;
    std::string testKey = "testKey";

    configuration.Set(testKey, 1);
    ASSERT_EQ(configuration.GetInt({testKey.c_str(), 0}), 0);
    configuration.AddIntConf({testKey, 0});
    configuration.Set(testKey, 1);
    ASSERT_EQ(configuration.GetInt({testKey.c_str(), 0}), 1);

    configuration.Set(testKey, 1.0f);
    ASSERT_EQ(configuration.GetFloat({testKey.c_str(), 0.0f}), 0.0f);
    configuration.mFloatItems[testKey] = 0.0;
    configuration.Set(testKey, 1.0f);
    ASSERT_EQ(configuration.GetFloat({testKey.c_str(), 0.0f}), 1.0f);

    configuration.Set(testKey, testKey);
    ASSERT_EQ(configuration.GetString({testKey.c_str(), ""}), "");
    configuration.AddStrConf({testKey, ""});
    configuration.Set(testKey, testKey);
    ASSERT_EQ(configuration.GetString({testKey.c_str(), testKey.c_str()}), testKey);

    configuration.Set(testKey, true);
    ASSERT_EQ(configuration.GetBool({testKey.c_str(), false}), false);
    configuration.AddBoolConf({testKey, false});
    configuration.Set(testKey, true);
    ASSERT_EQ(configuration.GetBool({testKey.c_str(), false}), true);

    configuration.Set(testKey, 1LU);
    ASSERT_EQ(configuration.GetUInt64({testKey.c_str(), 0LU}), 0LU);
    configuration.AddUInt64Conf({testKey, 0LU});
    configuration.Set(testKey, 1LU);
    ASSERT_EQ(configuration.GetUInt64({testKey.c_str(), 0LU}), 1LU);
}

TEST_F(TestMmcConfiguration, ParseMemSizeTest)
{
    Configuration configuration;
    auto ret = configuration.ParseMemSize("");
    ASSERT_EQ(ret, UINT64_MAX);

    ret = configuration.ParseMemSize("aB");
    ASSERT_EQ(ret, UINT64_MAX);

    ret = configuration.ParseMemSize("1B");
    ASSERT_EQ(ret, 1);

    ret = configuration.ParseMemSize("1KB");
    ASSERT_EQ(ret, KB_MEM_BYTES);

    ret = configuration.ParseMemSize("1MB");
    ASSERT_EQ(ret, MB_MEM_BYTES);

    ret = configuration.ParseMemSize("1GB");
    ASSERT_EQ(ret, GB_MEM_BYTES);

    ret = configuration.ParseMemSize("1TB");
    ASSERT_EQ(ret, TB_MEM_BYTES);
}

TEST_F(TestMmcConfiguration, MissingMaxDramSizeUsesDefault)
{
    ClientConfig clientConfig;
    clientConfig.LoadConfigurations();

    ASSERT_TRUE(clientConfig.Initialized());
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_MAX_DRAM_SIZE), "1TB");
}

TEST_F(TestMmcConfiguration, SetupWithErrorHandlingTest)
{
    ClientConfig clientConfig;
    auto ret = clientConfig.Setup(nullptr);
    ASSERT_FALSE(ret);

    auto config = CreateLocalConfigWithCurrentDefaults();
    SafeCopy("/non/exist/path.conf", config.config_path, sizeof(config.config_path));
    ret = clientConfig.Setup(&config);
    ASSERT_FALSE(ret);
}

TEST_F(TestMmcConfiguration, SetupWithFullConfigTest)
{
    ClientConfig clientConfig;
    auto config = CreateLocalConfigWithCurrentDefaults();
    SafeCopy("tcp://127.0.0.1:5001", config.meta_service_url, sizeof(config.meta_service_url));
    SafeCopy("tcp://127.0.0.1:6001", config.config_store_url, sizeof(config.config_store_url));
    SafeCopy("debug", config.log_level, sizeof(config.log_level));
    config.world_size = 128UL;
    SafeCopy("device_rdma", config.protocol, sizeof(config.protocol));
    SafeCopy("tcp://127.0.0.1:7001", config.hcom_url, sizeof(config.hcom_url));
    SafeCopy("2GB", config.dram_size, sizeof(config.dram_size));
    SafeCopy("4GB", config.hbm_size, sizeof(config.hbm_size));
    SafeCopy("128GB", config.max_dram_size, sizeof(config.max_dram_size));
    SafeCopy("64GB", config.max_hbm_size, sizeof(config.max_hbm_size));
    config.client_retry_milliseconds = 5000UL;
    config.client_timeout_seconds = 120UL;
    config.read_thread_pool_size = 16UL;
    config.write_thread_pool_size = 8UL;
    config.aggregate_io = true;
    config.aggregate_num = 256UL;

    config.tls_enable = true;
    SafeCopy("/etc/ssl/ca.pem", config.tls_ca_path, sizeof(config.tls_ca_path));
    SafeCopy("/etc/ssl/cert.pem", config.tls_cert_path, sizeof(config.tls_cert_path));
    SafeCopy("/etc/ssl/key.pem", config.tls_key_path, sizeof(config.tls_key_path));
    config.config_store_tls_enable = true;
    SafeCopy("/etc/ssl/cs_ca.pem", config.config_store_tls_ca_path, sizeof(config.config_store_tls_ca_path));
    config.hcom_tls_enable = true;
    SafeCopy("/etc/ssl/hcom_cert.pem", config.hcom_tls_cert_path, sizeof(config.hcom_tls_cert_path));

    const auto ret = clientConfig.Setup(&config);
    ASSERT_TRUE(ret);

    ASSERT_EQ(clientConfig.GetString(ConfConstant::OCK_MMC_META_SERVICE_URL), "tcp://127.0.0.1:5001");
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_BM_IP_PORT), "tcp://127.0.0.1:6001");
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OCK_MMC_LOG_LEVEL), "debug");
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OKC_MMC_LOCAL_SERVICE_WORLD_SIZE), 128UL);
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_PROTOCOL), "device_rdma");
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_BM_HCOM_URL), "tcp://127.0.0.1:7001");
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_DRAM_SIZE), "2GB");
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_HBM_SIZE), "4GB");
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_MAX_DRAM_SIZE), "128GB");
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OKC_MMC_LOCAL_SERVICE_MAX_HBM_SIZE), "64GB");
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OKC_MMC_CLIENT_RETRY_MILLISECONDS), 5000UL);
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OCK_MMC_CLIENT_TIMEOUT_SECONDS), 120UL);
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OCK_MMC_CLIENT_READ_THREAD_POOL_SIZE), 16UL);
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OCK_MMC_CLIENT_WRITE_THREAD_POOL_SIZE), 8UL);
    ASSERT_EQ(clientConfig.GetBool(ConfConstant::OCK_MMC_CLIENT_AGGREGATE_IO), true);
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OCK_MMC_CLIENT_AGGREGATE_NUM), 256UL);

    ASSERT_EQ(clientConfig.GetBool(ConfConstant::OCK_MMC_TLS_ENABLE), true);
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OCK_MMC_TLS_CA_PATH), "/etc/ssl/ca.pem");
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OCK_MMC_TLS_CERT_PATH), "/etc/ssl/cert.pem");
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OCK_MMC_TLS_KEY_PATH), "/etc/ssl/key.pem");
    ASSERT_EQ(clientConfig.GetBool(ConfConstant::OCK_MMC_CS_TLS_ENABLE), true);
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OCK_MMC_CS_TLS_CA_PATH), "/etc/ssl/cs_ca.pem");
    ASSERT_EQ(clientConfig.GetBool(ConfConstant::OCK_MMC_HCOM_TLS_ENABLE), true);
    ASSERT_EQ(clientConfig.GetString(ConfConstant::OCK_MMC_HCOM_TLS_CERT_PATH), "/etc/ssl/hcom_cert.pem");
}

TEST_F(TestMmcConfiguration, SetupWithBoundaryValuesTest)
{
    ClientConfig clientConfig;
    auto config = CreateLocalConfigWithCurrentDefaults();

    config.client_retry_milliseconds = MIN_RETRY_MS;
    config.client_timeout_seconds = MIN_TIMEOUT_SEC;
    config.read_thread_pool_size = MIN_THREAD_POOL_SIZE;
    config.write_thread_pool_size = MIN_THREAD_POOL_SIZE;
    config.aggregate_num = 1UL;

    auto ret = clientConfig.Setup(&config);
    ASSERT_TRUE(ret);

    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OKC_MMC_CLIENT_RETRY_MILLISECONDS), MIN_RETRY_MS);
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OCK_MMC_CLIENT_TIMEOUT_SECONDS), MIN_TIMEOUT_SEC);
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OCK_MMC_CLIENT_READ_THREAD_POOL_SIZE), MIN_THREAD_POOL_SIZE);
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OCK_MMC_CLIENT_WRITE_THREAD_POOL_SIZE), MIN_THREAD_POOL_SIZE);
    ASSERT_EQ(clientConfig.GetInt(ConfConstant::OCK_MMC_CLIENT_AGGREGATE_NUM), 1UL);

    ClientConfig clientConfigMax;
    config.client_retry_milliseconds = MAX_RETRY_MS;
    config.client_timeout_seconds = MAX_TIMEOUT_SEC;
    config.read_thread_pool_size = MAX_THREAD_POOL_SIZE;
    config.write_thread_pool_size = MAX_THREAD_POOL_SIZE;
    config.aggregate_num = MAX_AGGREGATE_NUM;
    config.world_size = MAX_WORLD_SIZE;

    ret = clientConfigMax.Setup(&config);
    ASSERT_TRUE(ret);

    ASSERT_EQ(clientConfigMax.GetInt(ConfConstant::OKC_MMC_CLIENT_RETRY_MILLISECONDS), MAX_RETRY_MS);
    ASSERT_EQ(clientConfigMax.GetInt(ConfConstant::OCK_MMC_CLIENT_TIMEOUT_SECONDS), MAX_TIMEOUT_SEC);
    ASSERT_EQ(clientConfigMax.GetInt(ConfConstant::OCK_MMC_CLIENT_READ_THREAD_POOL_SIZE), MAX_THREAD_POOL_SIZE);
    ASSERT_EQ(clientConfigMax.GetInt(ConfConstant::OCK_MMC_CLIENT_WRITE_THREAD_POOL_SIZE), MAX_THREAD_POOL_SIZE);
    ASSERT_EQ(clientConfigMax.GetInt(ConfConstant::OCK_MMC_CLIENT_AGGREGATE_NUM), MAX_AGGREGATE_NUM);
    ASSERT_EQ(clientConfigMax.GetInt(ConfConstant::OKC_MMC_LOCAL_SERVICE_WORLD_SIZE), MAX_WORLD_SIZE);
}

TEST_F(TestMmcConfiguration, DramBestEffortDefaultDisabledTest)
{
    ClientConfig clientConfig;
    auto config = CreateLocalConfigWithCurrentDefaults();
    const auto ret = clientConfig.Setup(&config);
    ASSERT_TRUE(ret);

    mmc_local_service_config_t localServiceConfig{};
    clientConfig.GetLocalServiceConfig(localServiceConfig);
    ASSERT_EQ(localServiceConfig.flags & SMEM_BM_FLAG_DRAM_BEST_EFFORT, 0U);
}

TEST_F(TestMmcConfiguration, DramBestEffortEnabledTest)
{
    ClientConfig clientConfig;
    auto config = CreateLocalConfigWithCurrentDefaults();
    auto ret = clientConfig.Setup(&config);
    ASSERT_TRUE(ret);

    const auto key = std::string(ConfConstant::OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT.first);
    clientConfig.Set(key, true);

    mmc_local_service_config_t localServiceConfig{};
    clientConfig.GetLocalServiceConfig(localServiceConfig);
    ASSERT_NE(localServiceConfig.flags & SMEM_BM_FLAG_DRAM_BEST_EFFORT, 0U);
}

TEST_F(TestMmcConfiguration, LogOutputTargetEnumConversionTest)
{
    // Test string-to-enum conversion for log_output_target config
    MetaServiceConfig metaConfig;
    metaConfig.LoadDefault();

    // NOTE: must use std::string() wrapper for the value argument — otherwise
    // Set(const string&, bool) is selected (const char* → bool is a standard
    // conversion, preferred over const char* → std::string user-defined).

    // "screen" -> LOG_OUTPUT_TARGET_SCREEN (0)
    metaConfig.Set(ConfConstant::OCK_MMC_LOG_OUTPUT_TARGET.first, std::string("screen"));
    mmc_meta_service_config_t config1{};
    metaConfig.GetMetaServiceConfig(config1);
    ASSERT_EQ(config1.logOutputTarget, LOG_OUTPUT_TARGET_SCREEN);

    // "file" -> LOG_OUTPUT_TARGET_FILE (1)
    metaConfig.Set(ConfConstant::OCK_MMC_LOG_OUTPUT_TARGET.first, std::string("file"));
    mmc_meta_service_config_t config2{};
    metaConfig.GetMetaServiceConfig(config2);
    ASSERT_EQ(config2.logOutputTarget, LOG_OUTPUT_TARGET_FILE);

    // "both" -> LOG_OUTPUT_TARGET_BOTH (2)
    metaConfig.Set(ConfConstant::OCK_MMC_LOG_OUTPUT_TARGET.first, std::string("both"));
    mmc_meta_service_config_t config3{};
    metaConfig.GetMetaServiceConfig(config3);
    ASSERT_EQ(config3.logOutputTarget, LOG_OUTPUT_TARGET_BOTH);

    // Default value (not explicitly set) should be "file" -> 1
    MetaServiceConfig defaultConfig;
    defaultConfig.LoadDefault();
    mmc_meta_service_config_t config4{};
    defaultConfig.GetMetaServiceConfig(config4);
    ASSERT_EQ(config4.logOutputTarget, LOG_OUTPUT_TARGET_FILE);

    // Case insensitivity: "SCREEN" and "Both" should work
    metaConfig.Set(ConfConstant::OCK_MMC_LOG_OUTPUT_TARGET.first, std::string("SCREEN"));
    mmc_meta_service_config_t config5{};
    metaConfig.GetMetaServiceConfig(config5);
    ASSERT_EQ(config5.logOutputTarget, LOG_OUTPUT_TARGET_SCREEN);

    metaConfig.Set(ConfConstant::OCK_MMC_LOG_OUTPUT_TARGET.first, std::string("Both"));
    mmc_meta_service_config_t config6{};
    metaConfig.GetMetaServiceConfig(config6);
    ASSERT_EQ(config6.logOutputTarget, LOG_OUTPUT_TARGET_BOTH);
}

TEST_F(TestMmcConfiguration, EvictCheckIntervalDefaultsAndCanBeDisabled)
{
    MetaServiceConfig metaConfig;
    metaConfig.LoadDefault();

    mmc_meta_service_config_t defaultConfig{};
    metaConfig.GetMetaServiceConfig(defaultConfig);
    EXPECT_EQ(defaultConfig.evictCheckIntervalSeconds, 5U);

    metaConfig.Set(ConfConstant::OCK_MMC_EVICT_CHECK_INTERVAL_SECONDS.first, static_cast<uint64_t>(0));
    mmc_meta_service_config_t disabledConfig{};
    metaConfig.GetMetaServiceConfig(disabledConfig);
    EXPECT_EQ(disabledConfig.evictCheckIntervalSeconds, 0U);
}

TEST_F(TestMmcConfiguration, DynamicConfigDefaultsInLocalServiceConfig)
{
    ClientConfig clientConfig;
    auto config = CreateLocalConfigWithCurrentDefaults();
    const auto ret = clientConfig.Setup(&config);
    ASSERT_TRUE(ret);

    mmc_local_service_config_t serviceConfig{};
    clientConfig.GetLocalServiceConfig(serviceConfig);

    EXPECT_TRUE(serviceConfig.dynamicConfigEnable);
    EXPECT_EQ(serviceConfig.dynamicConfigInterval, DEFAULT_DYNAMIC_CONFIG_INTERVAL);
}

TEST_F(TestMmcConfiguration, DynamicConfigSetupWithCustomValues)
{
    ClientConfig clientConfig;
    auto config = CreateLocalConfigWithCurrentDefaults();

    config.dynamic_config_enable = false;
    config.dynamic_config_interval = 10;

    const auto ret = clientConfig.Setup(&config);
    ASSERT_TRUE(ret);

    EXPECT_FALSE(clientConfig.GetBool(ConfConstant::OCK_MMC_DYNAMIC_CONFIG_ENABLE));
    EXPECT_EQ(clientConfig.GetInt(ConfConstant::OCK_MMC_DYNAMIC_CONFIG_INTERVAL), 10);
}

TEST_F(TestMmcConfiguration, DynamicConfigGetLocalServiceConfig)
{
    ClientConfig clientConfig;
    auto config = CreateLocalConfigWithCurrentDefaults();

    config.dynamic_config_enable = false;
    config.dynamic_config_interval = 15;

    const auto ret = clientConfig.Setup(&config);
    ASSERT_TRUE(ret);

    mmc_local_service_config_t serviceConfig{};
    clientConfig.GetLocalServiceConfig(serviceConfig);

    EXPECT_FALSE(serviceConfig.dynamicConfigEnable);
    EXPECT_EQ(serviceConfig.dynamicConfigInterval, 15U);
}

TEST_F(TestMmcConfiguration, DynamicConfigIntervalRangeBounds)
{
    EXPECT_EQ(MIN_DYNAMIC_CONFIG_INTERVAL, 1);
    EXPECT_EQ(MAX_DYNAMIC_CONFIG_INTERVAL, 300);
    EXPECT_EQ(DEFAULT_DYNAMIC_CONFIG_INTERVAL, 5);
}

namespace {
constexpr uint64_t HUGE_MEM_SIZE = 1ULL << 50; // 远超单机 DRAM/HBM 上限，用于触发 exceeds 校验
constexpr uint64_t ALIGNED_2MB = 2ULL * 1024 * 1024;
constexpr uint64_t ALIGNED_4MB = 4ULL * 1024 * 1024;

mmc_local_service_config_t MakeValidBaseConfig()
{
    mmc_local_service_config_t config{};
    config.logLevel = INFO_LEVEL;
    SafeCopy("host_rdma", config.dataOpType, PROTOCOL_SIZE);
    config.accTlsConfig.tlsEnable = false;
    config.hcomTlsConfig.tlsEnable = false;
    config.configStoreTlsConfig.tlsEnable = false;
    return config;
}
} // namespace

// localDRAMSize 超过上限
TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigDramExceedsMax_Fail)
{
    auto config = MakeValidBaseConfig();
    config.localDRAMSize = HUGE_MEM_SIZE;
    config.localMaxDRAMSize = ALIGNED_2MB;
    config.localHBMSize = ALIGNED_2MB;
    config.localMaxHBMSize = ALIGNED_2MB;
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

// localMaxDRAMSize 超过上限
TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigMaxDramExceedsMax_Fail)
{
    auto config = MakeValidBaseConfig();
    config.localDRAMSize = ALIGNED_2MB;
    config.localMaxDRAMSize = HUGE_MEM_SIZE;
    config.localHBMSize = ALIGNED_2MB;
    config.localMaxHBMSize = ALIGNED_2MB;
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

// localMaxDRAMSize 小于 localDRAMSize
TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigMaxDramLessThanDram_Fail)
{
    auto config = MakeValidBaseConfig();
    config.localDRAMSize = ALIGNED_4MB;
    config.localMaxDRAMSize = ALIGNED_2MB;
    config.localHBMSize = ALIGNED_2MB;
    config.localMaxHBMSize = ALIGNED_2MB;
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

// localHBMSize 超过上限
TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigHbmExceedsMax_Fail)
{
    auto config = MakeValidBaseConfig();
    config.localDRAMSize = ALIGNED_2MB;
    config.localMaxDRAMSize = ALIGNED_2MB;
    config.localHBMSize = HUGE_MEM_SIZE;
    config.localMaxHBMSize = ALIGNED_2MB;
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

// localMaxHBMSize 超过上限
TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigMaxHbmExceedsMax_Fail)
{
    auto config = MakeValidBaseConfig();
    config.localDRAMSize = ALIGNED_2MB;
    config.localMaxDRAMSize = ALIGNED_2MB;
    config.localHBMSize = ALIGNED_2MB;
    config.localMaxHBMSize = HUGE_MEM_SIZE;
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

// localMaxHBMSize 小于 localHBMSize
TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigMaxHbmLessThanHbm_Fail)
{
    auto config = MakeValidBaseConfig();
    config.localDRAMSize = ALIGNED_2MB;
    config.localMaxDRAMSize = ALIGNED_2MB;
    config.localHBMSize = ALIGNED_4MB;
    config.localMaxHBMSize = ALIGNED_2MB;
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

// acc_link TLS 配置无效
TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigTlsAccInvalid_Fail)
{
    auto config = MakeValidBaseConfig();
    config.localDRAMSize = ALIGNED_2MB;
    config.localMaxDRAMSize = ALIGNED_2MB;
    config.localHBMSize = ALIGNED_2MB;
    config.localMaxHBMSize = ALIGNED_2MB;
    config.accTlsConfig.tlsEnable = true; // 未提供有效路径
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

// hcom TLS 配置无效
TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigTlsHcomInvalid_Fail)
{
    auto config = MakeValidBaseConfig();
    config.localDRAMSize = ALIGNED_2MB;
    config.localMaxDRAMSize = ALIGNED_2MB;
    config.localHBMSize = ALIGNED_2MB;
    config.localMaxHBMSize = ALIGNED_2MB;
    config.hcomTlsConfig.tlsEnable = true;
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

// config store TLS 配置无效
TEST_F(TestMmcConfiguration, ValidateLocalServiceConfigTlsConfigStoreInvalid_Fail)
{
    auto config = MakeValidBaseConfig();
    config.localDRAMSize = ALIGNED_2MB;
    config.localMaxDRAMSize = ALIGNED_2MB;
    config.localHBMSize = ALIGNED_2MB;
    config.localMaxHBMSize = ALIGNED_2MB;
    config.configStoreTlsConfig.tlsEnable = true;
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

// 调用 GetClientConfig 覆盖其内部 getter 调用
TEST_F(TestMmcConfiguration, GetClientConfig_PopulatesConfig)
{
    ClientConfig clientConfig;
    auto config = CreateLocalConfigWithCurrentDefaults();
    ASSERT_TRUE(clientConfig.Setup(&config));
    mmc_client_config_t clientCfg{};
    clientConfig.GetClientConfig(clientCfg);
    EXPECT_EQ(string(clientCfg.dataOpType), "host_rdma");
    EXPECT_EQ(clientCfg.rpcRetryTimeOut, 0U);
    EXPECT_EQ(clientCfg.timeOut, 60U);
}

// RankedOpMetrics 覆盖 per-rank 计数器与序列化分支
TEST_F(TestMmcConfiguration, RankedOpMetricsIncrementAndAppendBranches)
{
    RankedOpMetrics metrics;
    constexpr uint32_t validRank = 7;
    constexpr uint32_t invalidRank = UINT32_MAX;
    // 合法 rank：递增四类计数（首次创建 + 已存在共享读路径）
    metrics.IncrementRequest(validRank);
    metrics.IncrementSuccess(validRank);
    metrics.IncrementFailure(validRank);
    metrics.IncrementNotFound(validRank);
    metrics.IncrementRequest(validRank); // 再次命中已存在分支
    // 非法 rank 应被忽略
    metrics.IncrementRequest(invalidRank);
    metrics.IncrementSuccess(invalidRank);
    metrics.IncrementFailure(invalidRank);
    metrics.IncrementNotFound(invalidRank);
    // 仅 notFound 的 rank，覆盖 rv/sv/fv==0 分支
    constexpr uint32_t notFoundOnlyRank = 9;
    metrics.IncrementNotFound(notFoundOnlyRank);

    std::ostringstream oss;
    metrics.AppendPerRankToStream(oss, "req", "succ", "fail", "nf");
    EXPECT_FALSE(oss.str().empty());

    // 空 nfName 分支
    std::ostringstream ossEmptyNf;
    metrics.AppendPerRankToStream(ossEmptyNf, "req", "succ", "fail", "");
    EXPECT_FALSE(ossEmptyNf.str().empty());
}

// SimplePerRankCounter 覆盖递增/递减/查询分支
TEST_F(TestMmcConfiguration, SimplePerRankCounterIncrementDecrementBranches)
{
    SimplePerRankCounter counter;
    constexpr uint32_t validRank = 3;
    constexpr uint32_t invalidRank = UINT32_MAX;
    counter.Increment(validRank);
    counter.Increment(validRank, 5);
    counter.Decrement(validRank, 2);
    counter.Increment(invalidRank); // 被忽略
    counter.Decrement(invalidRank); // 被忽略
    const auto rankMap = counter.GetRankMap();
    EXPECT_EQ(rankMap.at(validRank), 4U); // 1 + 5 - 2
}


TEST_F(TestMmcConfiguration, DramUnrestrictedDefaultDisabled)
{
    ClientConfig client;
    auto input = CreateLocalConfigWithCurrentDefaults();
    ASSERT_TRUE(client.Setup(&input));
    mmc_local_service_config_t config{};
    client.GetLocalServiceConfig(config);
    EXPECT_EQ(config.flags & MMC_BM_FLAG_DRAM_UNRESTRICTED, 0U);
}

TEST_F(TestMmcConfiguration, DramUnrestrictedEnabledForSdma)
{
    ClientConfig client;
    auto input = CreateLocalConfigWithCurrentDefaults();
    ASSERT_TRUE(client.Setup(&input));
    client.Set(ConfConstant::OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED.first, true);
    client.Set(ConfConstant::OKC_MMC_LOCAL_SERVICE_PROTOCOL.first, std::string("device_sdma"));
    mmc_local_service_config_t config{};
    client.GetLocalServiceConfig(config);
    EXPECT_NE(config.flags & MMC_BM_FLAG_DRAM_UNRESTRICTED, 0U);
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_OK);
}

TEST_F(TestMmcConfiguration, DramUnrestrictedRejectsOtherProtocols)
{
    mmc_local_service_config_t config{};
    config.flags = MMC_BM_FLAG_DRAM_UNRESTRICTED;
    SafeCopy("host_rdma", config.dataOpType, sizeof(config.dataOpType));
    EXPECT_EQ(ClientConfig::ValidateLocalServiceConfig(config), MMC_INVALID_PARAM);
}

TEST_F(TestMmcConfiguration, DramUnrestrictedFalsePreservesBestEffort)
{
    ClientConfig client;
    auto input = CreateLocalConfigWithCurrentDefaults();
    ASSERT_TRUE(client.Setup(&input));
    client.Set(ConfConstant::OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED.first, false);
    client.Set(ConfConstant::OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT.first, true);
    mmc_local_service_config_t config{};
    client.GetLocalServiceConfig(config);
    EXPECT_EQ(config.flags, SMEM_BM_FLAG_DRAM_BEST_EFFORT);
}
