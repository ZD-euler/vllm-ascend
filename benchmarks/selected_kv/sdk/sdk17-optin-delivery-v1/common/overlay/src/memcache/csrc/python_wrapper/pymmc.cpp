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

#include "mmc.h"
#include "mmc_def.h"
#include "mmc_ptracer.h"
#include "mmc_meta_service_process.h"
#include "common/mmc_functions.h"
#include "smem_bm_def.h"
#include "mmc_helper.h"
#include "pymmc.h"

namespace py = pybind11;
using namespace ock::mmc;

void DefineMmcStructModule(py::module_ &m)
{
    // Define the KeyInfo class
    py::class_<KeyInfo, std::shared_ptr<KeyInfo>>(m, "KeyInfo", py::buffer_protocol())
        .def("size", &KeyInfo::Size)
        .def("loc_list", &KeyInfo::GetLocs)
        .def("type_list", &KeyInfo::GetTypes)
        .def("gva_list", &KeyInfo::GetGvas)
        .def("__str__", &KeyInfo::ToString)
        .def("__repr__", &KeyInfo::ToString);

    py::class_<ReplicateConfig>(m, "ReplicateConfig", R"pbdoc(
         Configuration for replica allocation policy.
     )pbdoc")
        .def(py::init<>(), R"pbdoc(
         Default constructor.
         Initializes:
           - preferredLocalServiceIDs = {}
           - replicaNum = 0
     )pbdoc")
        .def_readwrite("replicaNum", &ReplicateConfig::replicaNum, R"pbdoc(
         Less than or equal to 8.
     )pbdoc")
        .def_readwrite("preferredLocalServiceIDs", &ReplicateConfig::preferredLocalServiceIDs, R"pbdoc(
         List of instance IDs for forced storage. The values in the list must be unique,
         and the list size must be equal to replicaNum.
     )pbdoc");

    auto localConfigClass =
        py::class_<local_config>(m, "LocalConfig", R"pbdoc(
         Local configuration for memcache client.
     )pbdoc")
            .def(py::init([]() { return create_default_local_config(); }),
                 R"pbdoc(
             Default constructor. Initializes with default values.
         )pbdoc")
            .def_property(
                "meta_service_url", [](const local_config &cfg) { return std::string(cfg.meta_service_url); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.meta_service_url, sizeof(cfg.meta_service_url));
                },
                R"pbdoc(
             Meta service start-up url.
         )pbdoc")
            .def_property(
                "config_store_url", [](const local_config &cfg) { return std::string(cfg.config_store_url); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.config_store_url, sizeof(cfg.config_store_url));
                },
                R"pbdoc(
             Config store url.
         )pbdoc")
            .def_property(
                "backend_id", [](const local_config &cfg) { return std::string(cfg.backend_id); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.backend_id, sizeof(cfg.backend_id));
                },
                R"pbdoc(
             Backend identity for kv_event, e.g. pod IP.
         )pbdoc")
            .def_property(
                "log_level", [](const local_config &cfg) { return std::string(cfg.log_level); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.log_level, sizeof(cfg.log_level));
                },
                R"pbdoc(
             Log level: debug, info, warn, error.
         )pbdoc")
            .def_readwrite("world_size", &local_config::world_size, R"pbdoc(
             The maximum supported rank count.
         )pbdoc")
            .def_property(
                "protocol", [](const local_config &cfg) { return std::string(cfg.protocol); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.protocol, sizeof(cfg.protocol));
                },
                R"pbdoc(
             Data transfer protocol: host_rdma, host_urma, host_tcp, host_shm, device_rdma, device_urma,
             device_uboe, device_sdma.
         )pbdoc")
            .def_property(
                "tag", [](const local_config &cfg) { return std::string(cfg.tag); },
                [](local_config &cfg, const std::string &value) { SafeCopy(value, cfg.tag, sizeof(cfg.tag)); },
                R"pbdoc(
             Tag of this rank in MemFabric Big Memory, used with tag_op_info to control the
             data op type (protocol) between tag pairs, e.g. tag1.
         )pbdoc")
            .def_property(
                "tag_op_info", [](const local_config &cfg) { return std::string(cfg.tag_op_info); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.tag_op_info, sizeof(cfg.tag_op_info));
                },
                R"pbdoc(
             Data op type between tags, separated by commas, e.g. "tag1:HOST_TCP:tag2,tag2:DEVICE_RDMA:tag3".
             Op type whitelist: DEVICE_SDMA, DEVICE_RDMA, DEVICE_URMA, DEVICE_UBOE, HOST_RDMA,
             HOST_TCP, HOST_URMA, HOST_SHM, AIV_SDMA, DEVICE_MTE.
         )pbdoc")
            .def_property(
                "hcom_url", [](const local_config &cfg) { return std::string(cfg.hcom_url); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.hcom_url, sizeof(cfg.hcom_url));
                },
                R"pbdoc(
             HCOM URL for RDMA network.
         )pbdoc")
            .def_property(
                "dram_size", [](const local_config &cfg) { return std::string(cfg.dram_size); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.dram_size, sizeof(cfg.dram_size));
                },
                R"pbdoc(
             DRAM space usage, supports formats like 1GB, 2MB, etc.
         )pbdoc")
            .def_property(
                "hbm_size", [](const local_config &cfg) { return std::string(cfg.hbm_size); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.hbm_size, sizeof(cfg.hbm_size));
                },
                R"pbdoc(
             HBM space usage.
         )pbdoc")
            .def_property(
                "max_dram_size", [](const local_config &cfg) { return std::string(cfg.max_dram_size); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.max_dram_size, sizeof(cfg.max_dram_size));
                },
                R"pbdoc(
             The MAX size of dram_size in all local processes.
         )pbdoc")
            .def_property(
                "max_hbm_size", [](const local_config &cfg) { return std::string(cfg.max_hbm_size); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.max_hbm_size, sizeof(cfg.max_hbm_size));
                },
                R"pbdoc(
             The MAX size of hbm_size in all local processes.
         )pbdoc")
            .def_readwrite("client_retry_milliseconds", &local_config::client_retry_milliseconds, R"pbdoc(
             The total retry duration when client requests meta service.
         )pbdoc")
            .def_readwrite("client_timeout_seconds", &local_config::client_timeout_seconds, R"pbdoc(
             Client request timeout in seconds.
         )pbdoc")
            .def_readwrite("read_thread_pool_size", &local_config::read_thread_pool_size, R"pbdoc(
             Read thread pool size.
         )pbdoc")
            .def_readwrite("write_thread_pool_size", &local_config::write_thread_pool_size, R"pbdoc(
             Write thread pool size.
         )pbdoc")
            .def_readwrite("aggregate_io", &local_config::aggregate_io, R"pbdoc(
             Enable read/write aggregate.
         )pbdoc")
            .def_readwrite("aggregate_num", &local_config::aggregate_num, R"pbdoc(
             Aggregate number.
         )pbdoc")
            .def_readwrite("tls_enable", &local_config::tls_enable, R"pbdoc(
             Enable TLS for metaservice.
         )pbdoc")
            .def_property(
                "tls_ca_path", [](const local_config &cfg) { return std::string(cfg.tls_ca_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.tls_ca_path, sizeof(cfg.tls_ca_path));
                },
                R"pbdoc(
             TLS CA certificate path.
         )pbdoc")
            .def_property(
                "tls_ca_crl_path", [](const local_config &cfg) { return std::string(cfg.tls_ca_crl_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.tls_ca_crl_path, sizeof(cfg.tls_ca_crl_path));
                },
                R"pbdoc(
             TLS CA CRL path.
         )pbdoc")
            .def_property(
                "tls_cert_path", [](const local_config &cfg) { return std::string(cfg.tls_cert_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.tls_cert_path, sizeof(cfg.tls_cert_path));
                },
                R"pbdoc(
             TLS certificate path.
         )pbdoc")
            .def_property(
                "tls_key_path", [](const local_config &cfg) { return std::string(cfg.tls_key_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.tls_key_path, sizeof(cfg.tls_key_path));
                },
                R"pbdoc(
             TLS key path.
         )pbdoc")
            .def_property(
                "tls_key_pass_path", [](const local_config &cfg) { return std::string(cfg.tls_key_pass_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.tls_key_pass_path, sizeof(cfg.tls_key_pass_path));
                },
                R"pbdoc(
             TLS key passphrase path.
         )pbdoc")
            .def_property(
                "tls_package_path", [](const local_config &cfg) { return std::string(cfg.tls_package_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.tls_package_path, sizeof(cfg.tls_package_path));
                },
                R"pbdoc(
             TLS package path.
         )pbdoc")
            .def_property(
                "tls_decrypter_path", [](const local_config &cfg) { return std::string(cfg.tls_decrypter_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.tls_decrypter_path, sizeof(cfg.tls_decrypter_path));
                },
                R"pbdoc(
             TLS decrypter library path.
         )pbdoc")
            .def_readwrite("config_store_tls_enable", &local_config::config_store_tls_enable, R"pbdoc(
             Enable TLS for config_store.
         )pbdoc")
            .def_property(
                "config_store_tls_ca_path",
                [](const local_config &cfg) { return std::string(cfg.config_store_tls_ca_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.config_store_tls_ca_path, sizeof(cfg.config_store_tls_ca_path));
                },
                R"pbdoc(
             Config store TLS CA certificate path.
         )pbdoc")
            .def_property(
                "config_store_tls_ca_crl_path",
                [](const local_config &cfg) { return std::string(cfg.config_store_tls_ca_crl_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.config_store_tls_ca_crl_path, sizeof(cfg.config_store_tls_ca_crl_path));
                },
                R"pbdoc(
             Config store TLS CA CRL path.
         )pbdoc")
            .def_property(
                "config_store_tls_cert_path",
                [](const local_config &cfg) { return std::string(cfg.config_store_tls_cert_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.config_store_tls_cert_path, sizeof(cfg.config_store_tls_cert_path));
                },
                R"pbdoc(
             Config store TLS certificate path.
         )pbdoc")
            .def_property(
                "config_store_tls_key_path",
                [](const local_config &cfg) { return std::string(cfg.config_store_tls_key_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.config_store_tls_key_path, sizeof(cfg.config_store_tls_key_path));
                },
                R"pbdoc(
             Config store TLS key path.
         )pbdoc")
            .def_property(
                "config_store_tls_key_pass_path",
                [](const local_config &cfg) { return std::string(cfg.config_store_tls_key_pass_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.config_store_tls_key_pass_path, sizeof(cfg.config_store_tls_key_pass_path));
                },
                R"pbdoc(
             Config store TLS key passphrase path.
         )pbdoc")
            .def_property(
                "config_store_tls_package_path",
                [](const local_config &cfg) { return std::string(cfg.config_store_tls_package_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.config_store_tls_package_path, sizeof(cfg.config_store_tls_package_path));
                },
                R"pbdoc(
             Config store TLS package path.
         )pbdoc")
            .def_property(
                "config_store_tls_decrypter_path",
                [](const local_config &cfg) { return std::string(cfg.config_store_tls_decrypter_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.config_store_tls_decrypter_path, sizeof(cfg.config_store_tls_decrypter_path));
                },
                R"pbdoc(
             Config store TLS decrypter library path.
         )pbdoc")
            .def_readwrite("hcom_tls_enable", &local_config::hcom_tls_enable, R"pbdoc(
             Enable TLS for HCOM.
         )pbdoc")
            .def_property(
                "hcom_tls_ca_path", [](const local_config &cfg) { return std::string(cfg.hcom_tls_ca_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.hcom_tls_ca_path, sizeof(cfg.hcom_tls_ca_path));
                },
                R"pbdoc(
             HCOM TLS CA certificate path.
         )pbdoc")
            .def_property(
                "hcom_tls_ca_crl_path", [](const local_config &cfg) { return std::string(cfg.hcom_tls_ca_crl_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.hcom_tls_ca_crl_path, sizeof(cfg.hcom_tls_ca_crl_path));
                },
                R"pbdoc(
             HCOM TLS CA CRL path.
         )pbdoc")
            .def_property(
                "hcom_tls_cert_path", [](const local_config &cfg) { return std::string(cfg.hcom_tls_cert_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.hcom_tls_cert_path, sizeof(cfg.hcom_tls_cert_path));
                },
                R"pbdoc(
             HCOM TLS certificate path.
         )pbdoc")
            .def_property(
                "hcom_tls_key_path", [](const local_config &cfg) { return std::string(cfg.hcom_tls_key_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.hcom_tls_key_path, sizeof(cfg.hcom_tls_key_path));
                },
                R"pbdoc(
             HCOM TLS key path.
         )pbdoc")
            .def_property(
                "hcom_tls_key_pass_path",
                [](const local_config &cfg) { return std::string(cfg.hcom_tls_key_pass_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.hcom_tls_key_pass_path, sizeof(cfg.hcom_tls_key_pass_path));
                },
                R"pbdoc(
             HCOM TLS key passphrase path.
         )pbdoc")
            .def_property(
                "hcom_tls_decrypter_path",
                [](const local_config &cfg) { return std::string(cfg.hcom_tls_decrypter_path); },
                [](local_config &cfg, const std::string &value) {
                    SafeCopy(value, cfg.hcom_tls_decrypter_path, sizeof(cfg.hcom_tls_decrypter_path));
                },
                R"pbdoc(
             HCOM TLS decrypter library path.
         )pbdoc");

    localConfigClass
        .def("str", &local_config_to_string, R"pbdoc(
             Returns a string representation of all member variables.
         )pbdoc")
        .def("__str__", &local_config_to_string, R"pbdoc(
             Returns a string representation of all member variables.
         )pbdoc")
        .def("__repr__", &local_config_to_string, R"pbdoc(
             Returns a string representation of all member variables.
         )pbdoc");

    auto metaConfigClass =
        py::class_<mmc_meta_service_config_t>(m, "MetaConfig", R"pbdoc(
            Meta service configuration.
        )pbdoc")
            .def(py::init([]() { return create_default_meta_config(); }),
                 R"pbdoc(
                    Default constructor. Initializes with default values.
                )pbdoc")
            .def_property(
                "meta_service_url",
                [](const mmc_meta_service_config_t &config) { return std::string(config.discoveryURL); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.discoveryURL, DISCOVERY_URL_SIZE);
                },
                R"pbdoc(
                    Meta service start-up url.
                )pbdoc")
            .def_property(
                "config_store_url",
                [](const mmc_meta_service_config_t &config) { return std::string(config.configStoreURL); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.configStoreURL, DISCOVERY_URL_SIZE);
                },
                R"pbdoc(
                    Config store url.
                )pbdoc")
            .def_property(
                "metrics_url", [](const mmc_meta_service_config_t &config) { return std::string(config.httpURL); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.httpURL, DISCOVERY_URL_SIZE);
                },
                R"pbdoc(
                    The metrics HTTP service url.
                )pbdoc")
            .def_property(
                "ha_enable", [](const mmc_meta_service_config_t &config) { return config.haEnable; },
                [](mmc_meta_service_config_t &config, bool value) { config.haEnable = value; },
                R"pbdoc(
             Enable or disable high availability deployment.
                )pbdoc")
            .def_property(
                "log_level",
                [](const mmc_meta_service_config_t &config) { return MetaLogLevelToString(config.logLevel); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    config.logLevel = MetaLogLevelFromString(value);
                },
                R"pbdoc(
                    Log level: debug, info, warn, error.
                )pbdoc")
            .def_property(
                "log_path", [](const mmc_meta_service_config_t &config) { return std::string(config.logPath); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.logPath, PATH_MAX_SIZE);
                },
                R"pbdoc(
                    Log directory path.
                )pbdoc")
            .def_property(
                "log_rotation_file_size",
                [](const mmc_meta_service_config_t &config) { return config.logRotationFileSize / (1024 * 1024); },
                [](mmc_meta_service_config_t &config, int32_t value) {
                    config.logRotationFileSize = value * (1024 * 1024);
                },
                R"pbdoc(
                    Log rotation file size in MB.
                )pbdoc")
            .def_readwrite("log_rotation_file_count", &mmc_meta_service_config_t::logRotationFileCount,
                           R"pbdoc(
                    Log rotation file count.
                )pbdoc")
            .def_property(
                "log_output_target",
                [](const mmc_meta_service_config_t &config) {
                    return MetaLogOutputTargetToString(config.logOutputTarget);
                },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    config.logOutputTarget = MetaLogOutputTargetFromString(value);
                },
                R"pbdoc(
                    Log output target: screen (stdout only), file (rotating file only), both (stdout + file).
                )pbdoc")
            .def_readwrite("evict_threshold_high", &mmc_meta_service_config_t::evictThresholdHigh,
                           R"pbdoc(
                Eviction high threshold in percentage.
                )pbdoc")
            .def_readwrite("evict_threshold_low", &mmc_meta_service_config_t::evictThresholdLow,
                           R"pbdoc(
                    Eviction low threshold in percentage.
                )pbdoc")
            .def_readwrite("evict_check_interval_seconds", &mmc_meta_service_config_t::evictCheckIntervalSeconds,
                           R"pbdoc(
                    Periodic eviction check interval in seconds. Set to 0 to disable.
                )pbdoc")
            .def_readwrite("lease_ttl_ms", &mmc_meta_service_config_t::leaseTtlMs,
                           R"pbdoc(
                    Default read lease TTL in milliseconds.
                )pbdoc")
            .def_readwrite("evict_yield_interval", &mmc_meta_service_config_t::evictYieldInterval,
                           R"pbdoc(
                    Number of evicted objects between yields. 0 means yield disabled.
                )pbdoc")
            .def_readwrite("evict_yield_duration_us", &mmc_meta_service_config_t::evictYieldDurationUs,
                           R"pbdoc(
                    Sleep duration in microseconds during each yield. Default 100us.
                )pbdoc")
            .def_property(
                "kv_events_enable", [](const mmc_meta_service_config_t &config) { return config.kvEvents.enable; },
                [](mmc_meta_service_config_t &config, bool value) { config.kvEvents.enable = value; },
                R"pbdoc(
                    Enable KV cache event publisher (default false).
                )pbdoc")
            .def_property(
                "kv_events_hash_as_int",
                [](const mmc_meta_service_config_t &config) { return config.kvEvents.hashAsInt; },
                [](mmc_meta_service_config_t &config, bool value) { config.kvEvents.hashAsInt = value; },
                R"pbdoc(
                    Emit seq/block hash as low-64-bit int (true, default; aligns vLLM
                    VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES) or full hex string (false).
                )pbdoc")
            .def_property(
                "kv_events_endpoint",
                [](const mmc_meta_service_config_t &config) { return std::string(config.kvEvents.endpoint); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.kvEvents.endpoint, sizeof(config.kvEvents.endpoint));
                },
                R"pbdoc(
                    ZMQ PUB bind endpoint for KV events, e.g. tcp://0.0.0.0:5557.
                )pbdoc")
            .def_property(
                "kv_events_model_name",
                [](const mmc_meta_service_config_t &config) { return std::string(config.kvEvents.modelName); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.kvEvents.modelName, sizeof(config.kvEvents.modelName));
                },
                R"pbdoc(
                    Model name tag for published KV events.
                )pbdoc")
            .def_property(
                "kv_events_tenant_id",
                [](const mmc_meta_service_config_t &config) { return std::string(config.kvEvents.tenantId); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.kvEvents.tenantId, sizeof(config.kvEvents.tenantId));
                },
                R"pbdoc(
                    tenant_id for published KV events.
                )pbdoc")
            .def_property(
                "kv_events_block_size",
                [](const mmc_meta_service_config_t &config) { return config.kvEvents.blockSize; },
                [](mmc_meta_service_config_t &config, uint32_t value) { config.kvEvents.blockSize = value; },
                R"pbdoc(
                    block_size for published KV events (0 = omit).
                )pbdoc")
            .def_property(
                "kv_events_queue_capacity",
                [](const mmc_meta_service_config_t &config) { return config.kvEvents.queueCapacity; },
                [](mmc_meta_service_config_t &config, uint32_t value) { config.kvEvents.queueCapacity = value; },
                R"pbdoc(
                    Bounded queue capacity for async KV event publishing.
                )pbdoc")
            .def_property(
                "kv_events_transport_type",
                [](const mmc_meta_service_config_t &config) { return config.kvEvents.transportType; },
                [](mmc_meta_service_config_t &config, int32_t value) { config.kvEvents.transportType = value; },
                R"pbdoc(
                    Transport plugin type for KV events (KV_EVENT_TRANSPORT enum, see mmc_types.h).
                )pbdoc")
            .def_property(
                "zmq_enable_curve", [](const mmc_meta_service_config_t &config) { return config.zmq.enableCurve; },
                [](mmc_meta_service_config_t &config, bool value) { config.zmq.enableCurve = value; },
                R"pbdoc(
                    Enable ZMQ CURVE encryption for the zmq transport (default false).
                )pbdoc")
            .def_property(
                "zmq_public_key",
                [](const mmc_meta_service_config_t &config) { return std::string(config.zmq.publicKey); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.zmq.publicKey, sizeof(config.zmq.publicKey));
                },
                R"pbdoc(
                    ZMQ CURVE server public key (Z85, 40 chars). Subscribers use it to decrypt.
                )pbdoc")
            .def_property(
                "zmq_private_key", [](const mmc_meta_service_config_t &) { return std::string("***"); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.zmq.secretKey, sizeof(config.zmq.secretKey));
                },
                R"pbdoc(
                    ZMQ CURVE server secret key (Z85, 40 chars). Write-only: the
                    getter returns "***" to avoid leaking the private key material.
                )pbdoc")
            .def_property(
                "tls_enable", [](const mmc_meta_service_config_t &config) { return config.accTlsConfig.tlsEnable; },
                [](mmc_meta_service_config_t &config, bool value) { config.accTlsConfig.tlsEnable = value; },
                R"pbdoc(
                    Enable TLS for metaservice.
                )pbdoc")
            .def_property(
                "tls_ca_path",
                [](const mmc_meta_service_config_t &config) { return std::string(config.accTlsConfig.caPath); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.accTlsConfig.caPath, sizeof(config.accTlsConfig.caPath));
                },
                R"pbdoc(
                    TLS CA certificate path.
                )pbdoc")
            .def_property(
                "tls_ca_crl_path",
                [](const mmc_meta_service_config_t &config) { return std::string(config.accTlsConfig.crlPath); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.accTlsConfig.crlPath, sizeof(config.accTlsConfig.crlPath));
                },
                R"pbdoc(
                    TLS CA CRL path.
                )pbdoc")
            .def_property(
                "tls_cert_path",
                [](const mmc_meta_service_config_t &config) { return std::string(config.accTlsConfig.certPath); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.accTlsConfig.certPath, sizeof(config.accTlsConfig.certPath));
                },
                R"pbdoc(
             TLS certificate path.
                )pbdoc")
            .def_property(
                "tls_key_path",
                [](const mmc_meta_service_config_t &config) { return std::string(config.accTlsConfig.keyPath); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.accTlsConfig.keyPath, sizeof(config.accTlsConfig.keyPath));
                },
                R"pbdoc(
                    TLS key path.
                )pbdoc")
            .def_property(
                "tls_key_pass_path",
                [](const mmc_meta_service_config_t &config) { return std::string(config.accTlsConfig.keyPassPath); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.accTlsConfig.keyPassPath, sizeof(config.accTlsConfig.keyPassPath));
                },
                R"pbdoc(
             TLS key passphrase path.
                )pbdoc")
            .def_property(
                "tls_package_path",
                [](const mmc_meta_service_config_t &config) { return std::string(config.accTlsConfig.packagePath); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.accTlsConfig.packagePath, sizeof(config.accTlsConfig.packagePath));
                },
                R"pbdoc(
                    TLS package path.
                )pbdoc")
            .def_property(
                "tls_decrypter_path",
                [](const mmc_meta_service_config_t &config) {
                    return std::string(config.accTlsConfig.decrypterLibPath);
                },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.accTlsConfig.decrypterLibPath, sizeof(config.accTlsConfig.decrypterLibPath));
                },
                R"pbdoc(
                    TLS decrypter library path.
                )pbdoc")
            .def_property(
                "config_store_tls_enable",
                [](const mmc_meta_service_config_t &config) { return config.configStoreTlsConfig.tlsEnable; },
                [](mmc_meta_service_config_t &config, bool value) { config.configStoreTlsConfig.tlsEnable = value; },
                R"pbdoc(
                    Enable TLS for config store.
                )pbdoc")
            .def_property(
                "config_store_tls_ca_path",
                [](const mmc_meta_service_config_t &config) { return std::string(config.configStoreTlsConfig.caPath); },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.configStoreTlsConfig.caPath, sizeof(config.configStoreTlsConfig.caPath));
                },
                R"pbdoc(
                    Config store TLS CA certificate path.
                )pbdoc")
            .def_property(
                "config_store_tls_ca_crl_path",
                [](const mmc_meta_service_config_t &config) {
                    return std::string(config.configStoreTlsConfig.crlPath);
                },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.configStoreTlsConfig.crlPath, sizeof(config.configStoreTlsConfig.crlPath));
                },
                R"pbdoc(
                    Config store TLS CA CRL path.
                )pbdoc")
            .def_property(
                "config_store_tls_cert_path",
                [](const mmc_meta_service_config_t &config) {
                    return std::string(config.configStoreTlsConfig.certPath);
                },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.configStoreTlsConfig.certPath, sizeof(config.configStoreTlsConfig.certPath));
                },
                R"pbdoc(
                    Config store TLS certificate path.
                )pbdoc")
            .def_property(
                "config_store_tls_key_path",
                [](const mmc_meta_service_config_t &config) {
                    return std::string(config.configStoreTlsConfig.keyPath);
                },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.configStoreTlsConfig.keyPath, sizeof(config.configStoreTlsConfig.keyPath));
                },
                R"pbdoc(
                    Config store TLS key path.
                )pbdoc")
            .def_property(
                "config_store_tls_key_pass_path",
                [](const mmc_meta_service_config_t &config) {
                    return std::string(config.configStoreTlsConfig.keyPassPath);
                },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.configStoreTlsConfig.keyPassPath,
                             sizeof(config.configStoreTlsConfig.keyPassPath));
                },
                R"pbdoc(
                    Config store TLS key passphrase path.
                )pbdoc")
            .def_property(
                "config_store_tls_package_path",
                [](const mmc_meta_service_config_t &config) {
                    return std::string(config.configStoreTlsConfig.packagePath);
                },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.configStoreTlsConfig.packagePath,
                             sizeof(config.configStoreTlsConfig.packagePath));
                },
                R"pbdoc(
                    Config store TLS package path.
                )pbdoc")
            .def_property(
                "config_store_tls_decrypter_path",
                [](const mmc_meta_service_config_t &config) {
                    return std::string(config.configStoreTlsConfig.decrypterLibPath);
                },
                [](mmc_meta_service_config_t &config, const std::string &value) {
                    SafeCopy(value, config.configStoreTlsConfig.decrypterLibPath,
                             sizeof(config.configStoreTlsConfig.decrypterLibPath));
                },
                R"pbdoc(
                    Config store TLS decrypter library path.
                )pbdoc");

    metaConfigClass
        .def("str", &meta_config_to_string, R"pbdoc(
             Returns a string representation of all member variables.
         )pbdoc")
        .def("__str__", &meta_config_to_string, R"pbdoc(
             Returns a string representation of all member variables.
         )pbdoc")
        .def("__repr__", &meta_config_to_string, R"pbdoc(
             Returns a string representation of all member variables.
         )pbdoc");
}

PYBIND11_MODULE(_pymmc, m)
{
    DefineMmcStructModule(m);
    ock::mmc::ReplicateConfig defaultConfig;
    // Support starting the meta service from python
    py::class_<MmcMetaServiceProcess>(m, "MetaService", R"pbdoc(
         Class for memcache meta service process.
     )pbdoc")
        .def_static(
            "setup",
            [](const mmc_meta_service_config_t &config) { return MmcMetaServiceProcess::getInstance().Setup(config); },
            py::arg("config"), "Set meta service startup config.")
        .def_static(
            "main", []() { return MmcMetaServiceProcess::getInstance().MainForPython(); },
            py::call_guard<py::gil_scoped_release>(),
            "Start the meta service process directly. This is a blocking call.");

    // Define the MmcacheStore class
    py::class_<MmcacheStore>(m, "DistributedObjectStore")
        .def(py::init<>())
        .def("setup", &MmcacheStore::Setup, py::call_guard<py::gil_scoped_release>(), py::arg("config"),
             "Setup local configuration")
        .def("init", &MmcacheStore::Init, py::call_guard<py::gil_scoped_release>(), py::arg("device_id"),
             py::arg("init_bm") = true)
        .def("remove", &MmcacheStore::Remove, py::call_guard<py::gil_scoped_release>())
        .def("remove_batch", &MmcacheStore::BatchRemove, py::call_guard<py::gil_scoped_release>())
        .def("remove_all", &MmcacheStore::RemoveAll, py::call_guard<py::gil_scoped_release>())
        .def("is_exist", &MmcacheStore::IsExist, py::call_guard<py::gil_scoped_release>())
        .def("batch_is_exist", &MmcacheStore::BatchIsExist, py::call_guard<py::gil_scoped_release>(), py::arg("keys"),
             "Check if multiple objects exist. Returns list of results: 1 if exists, 0 if not exists, -1 if error")
        .def("get_key_info", &MmcacheStore::GetKeyInfo, py::call_guard<py::gil_scoped_release>(), py::arg("key"),
             py::arg("flag") = GET_KEY_INFO_FOR_LAYER_WISE)
        .def("batch_get_key_info", &MmcacheStore::BatchGetKeyInfo, py::call_guard<py::gil_scoped_release>(),
             py::arg("keys"), py::arg("flag") = GET_KEY_INFO_FOR_LAYER_WISE)
        .def("prefetch", &MmcacheStore::PrefetchKeys, py::call_guard<py::gil_scoped_release>(), py::arg("keys"),
             py::arg("src_media") = MMC_MEDIA_SSD, py::arg("dst_media") = MMC_MEDIA_DRAM, py::arg("flags") = 0,
             "Prefetch keys from src_media to dst_media. Only SSD->DRAM supported. Returns 0 on success.")
        .def("evict", &MmcacheStore::EvictKeys, py::call_guard<py::gil_scoped_release>(), py::arg("keys"),
             py::arg("src_media") = MMC_MEDIA_DRAM, py::arg("dst_media") = MMC_MEDIA_SSD, py::arg("flags") = 0,
             "Evict keys from src_media to dst_media. Only DRAM->SSD supported. Returns 0 on success.")
        .def("batch_add_lease", &MmcacheStore::BatchAddLease, py::call_guard<py::gil_scoped_release>(), py::arg("keys"),
             py::arg("leaseTtlMs") = 0)
        .def("batch_add_lease_with_info", &MmcacheStore::BatchAddLeaseWithInfo,
             py::call_guard<py::gil_scoped_release>(), py::arg("keys"), py::arg("lease_ttl_ms") = 0,
             "Acquire read leases once and return aligned (statuses, KeyInfos) from the same Add response")
        .def("batch_remove_lease", &MmcacheStore::BatchRemoveLease, py::call_guard<py::gil_scoped_release>(),
             py::arg("keys"))
        .def("close", &MmcacheStore::TearDown)
        .def(
            "register_buffer",
            [](MmcacheStore &self, uintptr_t buffer_ptr, size_t size) {
                // Register memory buffer for RDMA operations
                void *buffer = reinterpret_cast<void *>(buffer_ptr);
                py::gil_scoped_release release;
                return self.RegisterBuffer(buffer, size);
            },
            py::arg("buffer_ptr"), py::arg("size"), "Register a memory buffer for direct access operations")
        .def(
            "unregister_buffer",
            [](MmcacheStore &self, uintptr_t buffer_ptr, size_t size) {
                // UnRegister memory buffer for RDMA operations
                void *buffer = reinterpret_cast<void *>(buffer_ptr);
                py::gil_scoped_release release;
                return self.UnRegisterBuffer(buffer, size);
            },
            py::arg("buffer_ptr"), py::arg("size"), "UnRegister a memory buffer for direct access operations")
        .def(
            "get_into",
            [](MmcacheStore &self, const std::string &key, uintptr_t buffer_ptr, size_t size, const int32_t &direct) {
                py::gil_scoped_release release;
                return self.GetInto(key, reinterpret_cast<void *>(buffer_ptr), size, direct);
            },
            py::arg("key"), py::arg("buffer_ptr"), py::arg("size"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_G2H),
            "Get object data directly into a pre-allocated buffer")
        .def(
            "batch_get_into",
            [](MmcacheStore &self, const std::vector<std::string> &keys, const std::vector<uintptr_t> &buffer_ptrs,
               const std::vector<size_t> &sizes, const int32_t &direct) {
                std::vector<void *> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (uintptr_t ptr : buffer_ptrs) {
                    buffers.push_back(reinterpret_cast<void *>(ptr));
                }
                py::gil_scoped_release release;
                return self.BatchGetInto(keys, buffers, sizes, direct);
            },
            py::arg("keys"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_G2H),
            "Get object data directly into pre-allocated buffers for multiple "
            "keys")
        .def(
            "get_into_layers",
            [](MmcacheStore &self, const std::string &key, const std::vector<uintptr_t> &buffer_ptrs,
               const std::vector<size_t> &sizes, const int32_t &direct) {
                TP_TRACE_BEGIN(TP_MMC_PYBIND_GET_LAYERS);
                std::vector<void *> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (uintptr_t ptr : buffer_ptrs) {
                    buffers.push_back(reinterpret_cast<void *>(ptr));
                }
                py::gil_scoped_release release;
                auto ret = self.GetIntoLayers(key, buffers, sizes, direct);
                TP_TRACE_END(TP_MMC_PYBIND_GET_LAYERS, 0);
                return ret;
            },
            py::arg("key"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_G2H))
        .def(
            "batch_get_into_layers",
            [](MmcacheStore &self, const std::vector<std::string> &keys,
               const std::vector<std::vector<uintptr_t>> &buffer_ptrs, const std::vector<std::vector<size_t>> &sizes,
               const int32_t &direct) {
                TP_TRACE_BEGIN(TP_MMC_PYBIND_BATCH_GET_LAYERS);
                std::vector<std::vector<void *>> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (auto vec : buffer_ptrs) {
                    std::vector<void *> tmp;
                    for (uintptr_t ptr : vec) {
                        tmp.push_back(reinterpret_cast<void *>(ptr));
                    }
                    buffers.push_back(tmp);
                }
                py::gil_scoped_release release;
                auto ret = self.BatchGetIntoLayers(keys, buffers, sizes, direct);
                TP_TRACE_END(TP_MMC_PYBIND_BATCH_GET_LAYERS, 0);
                return ret;
            },
            py::arg("keys"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_G2H))
        .def(
            "get_local_service_id",
            [](MmcacheStore &self) {
                py::gil_scoped_release release;
                uint32_t localServiceId = std::numeric_limits<uint32_t>::max();
                self.GetLocalServiceId(localServiceId);
                return localServiceId;
            },
            "Get local serviceId")
        .def(
            "put_from",
            [](MmcacheStore &self, const std::string &key, uintptr_t buffer_ptr, size_t size, const int32_t &direct,
               const ReplicateConfig &replicateConfig) {
                py::gil_scoped_release release;
                return self.PutFrom(key, reinterpret_cast<void *>(buffer_ptr), size, direct, replicateConfig);
            },
            py::arg("key"), py::arg("buffer_ptr"), py::arg("size"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_H2G), py::arg("replicateConfig") = defaultConfig,
            "Put object data directly from a pre-allocated buffer")
        .def(
            "batch_put_from",
            [](MmcacheStore &self, const std::vector<std::string> &keys, const std::vector<uintptr_t> &buffer_ptrs,
               const std::vector<size_t> &sizes, const int32_t &direct, const ReplicateConfig &replicateConfig) {
                std::vector<void *> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (uintptr_t ptr : buffer_ptrs) {
                    buffers.push_back(reinterpret_cast<void *>(ptr));
                }
                py::gil_scoped_release release;
                return self.BatchPutFrom(keys, buffers, sizes, direct, replicateConfig);
            },
            py::arg("keys"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_H2G), py::arg("replicateConfig") = defaultConfig,
            "Put object data directly from pre-allocated buffers for multiple "
            "keys")
        .def(
            "put_from_layers",
            [](MmcacheStore &self, const std::string &key, const std::vector<uintptr_t> &buffer_ptrs,
               const std::vector<size_t> &sizes, const int32_t &direct, const ReplicateConfig &replicateConfig) {
                TP_TRACE_BEGIN(TP_MMC_PYBIND_PUT_LAYERS);
                std::vector<void *> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (uintptr_t ptr : buffer_ptrs) {
                    buffers.push_back(reinterpret_cast<void *>(ptr));
                }
                py::gil_scoped_release release;
                auto ret = self.PutFromLayers(key, buffers, sizes, direct, replicateConfig);
                TP_TRACE_END(TP_MMC_PYBIND_PUT_LAYERS, 0);
                return ret;
            },
            py::arg("key"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_H2G), py::arg("replicateConfig") = defaultConfig)
        .def(
            "batch_put_from_layers",
            [](MmcacheStore &self, const std::vector<std::string> &keys,
               const std::vector<std::vector<uintptr_t>> &buffer_ptrs, const std::vector<std::vector<size_t>> &sizes,
               const int32_t &direct, const ReplicateConfig &replicateConfig) {
                TP_TRACE_BEGIN(TP_MMC_PYBIND_BATCH_PUT_LAYERS);
                std::vector<std::vector<void *>> buffers;
                buffers.reserve(buffer_ptrs.size());
                for (auto vec : buffer_ptrs) {
                    std::vector<void *> tmp;
                    for (uintptr_t ptr : vec) {
                        tmp.push_back(reinterpret_cast<void *>(ptr));
                    }
                    buffers.push_back(tmp);
                }
                py::gil_scoped_release release;
                auto ret = self.BatchPutFromLayers(keys, buffers, sizes, direct, replicateConfig);
                TP_TRACE_END(TP_MMC_PYBIND_BATCH_PUT_LAYERS, 0);
                return ret;
            },
            py::arg("keys"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_H2G), py::arg("replicateConfig") = defaultConfig)
        .def(
            "batch_alloc",
            [](MmcacheStore &self, const std::vector<std::string> &keys, const std::vector<size_t> &sizes,
               uint16_t media, uint64_t leaseTtlMs) {
                py::gil_scoped_release release;
                return self.BatchMalloc(keys, sizes, media, leaseTtlMs);
            },
            py::arg("keys"), py::arg("sizes"), py::arg("media") = 1, py::arg("leaseTtlMs") = 0) // MEDIA_DRAM
        .def(
            "batch_copy",
            [](MmcacheStore &self, const std::vector<uintptr_t> &gva_ptrs, const std::vector<uintptr_t> &buffer_ptrs,
               std::vector<size_t> &sizes, const int32_t &direct) {
                if (gva_ptrs.size() != buffer_ptrs.size() || gva_ptrs.size() != sizes.size()) {
                    return -1;
                }
                std::vector<void *> gvaVec{};
                std::vector<void *> bufVec{};
                for (size_t i = 0; i < gva_ptrs.size(); i++) {
                    gvaVec.push_back(reinterpret_cast<void *>(gva_ptrs[i]));
                    bufVec.push_back(reinterpret_cast<void *>(buffer_ptrs[i]));
                }

                py::gil_scoped_release release;
                return self.BatchCopy(gvaVec, bufVec, sizes, direct);
            },
            py::arg("gva_ptrs"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_G2L))
        .def(
            "batch_copy_layers",
            [](MmcacheStore &self, const std::vector<uintptr_t> &gva_ptrs,
               const std::vector<std::vector<uintptr_t>> &buffer_ptrs, const std::vector<std::vector<size_t>> &sizes,
               const int32_t &direct) {
                if (gva_ptrs.size() != buffer_ptrs.size() || gva_ptrs.size() != sizes.size()) {
                    return -1;
                }

                std::vector<void *> gvaVec{};
                std::vector<void *> bufVec{};
                std::vector<size_t> sizeVec{};
                for (size_t i = 0; i < gva_ptrs.size(); i++) {
                    uintptr_t gva = gva_ptrs[i];
                    for (size_t j = 0; j < buffer_ptrs[i].size(); j++) {
                        auto &buffer = buffer_ptrs[i][j];
                        auto size = sizes[i][j];
                        sizeVec.push_back(size);
                        bufVec.push_back(reinterpret_cast<void *>(buffer));
                        gvaVec.push_back(reinterpret_cast<void *>(gva));
                        gva += size;
                    }
                }

                py::gil_scoped_release release;
                return self.BatchCopy(gvaVec, bufVec, sizeVec, direct);
            },
            py::arg("gva_ptrs"), py::arg("buffer_ptrs"), py::arg("sizes"),
            py::arg("direct") = static_cast<int32_t>(SMEMB_COPY_G2L))
        .def(
            "batch_write_finish",
            [](MmcacheStore &self, const std::vector<std::string> &keys, const std::vector<int32_t> &writeResults) {
                if (keys.size() != writeResults.size()) {
                    return std::vector<int>(keys.size(), -1);
                }
                py::gil_scoped_release release;
                return self.BatchWriteFinish(keys, writeResults);
            },
            py::arg("keys"), py::arg("res"),
            "Notify meta service that the writes for the given keys are finished. res is per-key "
            "write result (0=OK, non-zero=FAIL). Returns per-key meta service update result.")
        .def(
            "put",
            [](MmcacheStore &self, const std::string &key, const py::buffer &buf,
               const ReplicateConfig &replicateConfig) {
                py::buffer_info info = buf.request(false);
                mmc_buffer buffer = {.addr = reinterpret_cast<uint64_t>(info.ptr),
                                     .type = MEDIA_DRAM,
                                     .offset = 0,
                                     .len = static_cast<uint64_t>(info.size)};
                py::gil_scoped_release release;
                return self.Put(key, buffer, replicateConfig);
            },
            py::arg("key"), py::arg("buf"), py::arg("replicateConfig") = defaultConfig)
        .def(
            "put_batch",
            [](MmcacheStore &self, const std::vector<std::string> &keys, const std::vector<py::buffer> &buffers,
               const ReplicateConfig &replicateConfig) {
                // Convert pybuffers to spans without copying
                std::vector<py::buffer_info> infos;
                std::vector<mmc_buffer> bufs;
                infos.reserve(buffers.size());
                bufs.reserve(buffers.size());

                for (const auto &buf : buffers) {
                    infos.emplace_back(buf.request(false));
                    const auto &info = infos.back();
                    bufs.emplace_back(mmc_buffer{.addr = reinterpret_cast<uint64_t>(info.ptr),
                                                 .type = MEDIA_DRAM,
                                                 .offset = 0,
                                                 .len = static_cast<uint64_t>(info.size)});
                }
                py::gil_scoped_release release;
                return self.PutBatch(keys, bufs, replicateConfig);
            },
            py::arg("keys"), py::arg("values"), py::arg("replicateConfig") = defaultConfig)
        .def("get",
             [](MmcacheStore &self, const std::string &key) {
                 mmc_buffer buffer = self.Get(key);
                 py::gil_scoped_acquire acquire_gil;
                 if (buffer.addr == 0) {
                     return py::bytes("");
                 } else {
                     auto dataPtr = reinterpret_cast<char *>(buffer.addr);
                     std::shared_ptr<char[]> dataSharedPtr(dataPtr, [](char *p) { delete[] p; });
                     return py::bytes(dataPtr, buffer.len);
                 }
             })
        .def("get_batch", [](MmcacheStore &self, const std::vector<std::string> &keys) {
            const auto kNullString = pybind11::bytes("\\0", 0);
            {
                py::gil_scoped_release release_gil;
                auto bufferArray = self.GetBatch(keys);

                py::gil_scoped_acquire acquire_gil;
                std::vector<pybind11::bytes> results;
                results.reserve(keys.size());

                for (size_t i = 0; i < keys.size(); ++i) {
                    if (bufferArray.empty()) {
                        results.emplace_back(kNullString);
                        continue;
                    }
                    mmc_buffer buffer = bufferArray[i];
                    if (buffer.addr == 0) {
                        results.emplace_back(kNullString);
                        continue;
                    }
                    auto dataPtr = reinterpret_cast<char *>(buffer.addr);
                    std::shared_ptr<char[]> dataSharedPtr(dataPtr, [](char *p) { delete[] p; });
                    results.emplace_back(py::bytes(dataPtr, buffer.len));
                }
                return results;
            }
        });
}
