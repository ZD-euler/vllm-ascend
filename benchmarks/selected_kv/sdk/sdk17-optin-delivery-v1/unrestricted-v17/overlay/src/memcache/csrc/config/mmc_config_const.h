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
#ifndef MEM_FABRIC_MMC_CONFIG_CONST_H
#define MEM_FABRIC_MMC_CONFIG_CONST_H

#include <utility>
#include <cstdint>

constexpr uint32_t DEFAULT_DYNAMIC_CONFIG_INTERVAL = 5;

namespace ock {
namespace mmc {
namespace ConfConstant {
// add configuration here with default values
constexpr auto OCK_MMC_META_SERVICE_URL = std::make_pair("ock.mmc.meta_service_url", "tcp://127.0.0.1:5000");
constexpr auto OCK_MMC_META_SERVICE_CONFIG_STORE_URL =
    std::make_pair("ock.mmc.meta_service.config_store_url", "tcp://127.0.0.1:6000");
constexpr auto OCK_MMC_META_SERVICE_HTTP_URL =
    std::make_pair("ock.mmc.meta_service.metrics_url", "http://127.0.0.1:8000");
constexpr auto OCK_MMC_METRICS_REPORT_INTERVAL_SECONDS =
    std::make_pair("ock.mmc.meta_service.metrics_report_interval_seconds", 30UL);
constexpr auto OCK_MMC_META_HA_ENABLE = std::make_pair("ock.mmc.meta.ha.enable", false);
constexpr auto OKC_MMC_EVICT_THRESHOLD_HIGH = std::make_pair("ock.mmc.evict_threshold_high", 90);
constexpr auto OKC_MMC_EVICT_THRESHOLD_LOW = std::make_pair("ock.mmc.evict_threshold_low", 80);
constexpr auto OCK_MMC_EVICT_CHECK_INTERVAL_SECONDS = std::make_pair("ock.mmc.evict_check_interval_seconds", 5);
constexpr auto OCK_MMC_META_LEASE_TTL_MS = std::make_pair("ock.mmc.meta.lease_ttl_ms", 10000);
constexpr auto OKC_MMC_LOCAL_SERVICE_BACKEND_ID = std::make_pair("ock.mmc.local_service.backend_id", "");
// GC eviction yield: periodically sleep during eviction loop to release the write lock,
// allowing blocked Put/Get/Exist requests to acquire the lock and execute.
// yieldInterval=0 disables yield (backward compatible). Recommended: interval=100, duration=100us.
constexpr auto OKC_MMC_EVICT_YIELD_INTERVAL = std::make_pair("ock.mmc.evict_yield_interval", 0);
constexpr auto OKC_MMC_EVICT_YIELD_DURATION_US = std::make_pair("ock.mmc.evict_yield_duration_us", 100);
constexpr auto OCK_MMC_LOG_LEVEL = std::make_pair("ock.mmc.log_level", "info");
constexpr auto OCK_MMC_LOG_PATH = std::make_pair("ock.mmc.log_path", "/var/log/memcache_hybrid");
constexpr auto OCK_MMC_LOG_ROTATION_FILE_SIZE = std::make_pair("ock.mmc.log_rotation_file_size", 20);
constexpr auto OCK_MMC_LOG_ROTATION_FILE_COUNT = std::make_pair("ock.mmc.log_rotation_file_count", 50);
constexpr auto OCK_MMC_LOG_OUTPUT_TARGET = std::make_pair("ock.mmc.log_output_target", "file");

constexpr auto OCK_MMC_TLS_ENABLE = std::make_pair("ock.mmc.tls.enable", false);
constexpr auto OCK_MMC_TLS_CA_PATH = std::make_pair("ock.mmc.tls.ca.path", "");
constexpr auto OCK_MMC_TLS_CRL_PATH = std::make_pair("ock.mmc.tls.ca.crl.path", "");
constexpr auto OCK_MMC_TLS_CERT_PATH = std::make_pair("ock.mmc.tls.cert.path", "");
constexpr auto OCK_MMC_TLS_KEY_PATH = std::make_pair("ock.mmc.tls.key.path", "");
constexpr auto OCK_MMC_TLS_KEY_PASS_PATH = std::make_pair("ock.mmc.tls.key.pass.path", "");
constexpr auto OCK_MMC_TLS_PACKAGE_PATH = std::make_pair("ock.mmc.tls.package.path", "");
constexpr auto OCK_MMC_TLS_DECRYPTER_PATH = std::make_pair("ock.mmc.tls.decrypter.path", "");
constexpr auto OCK_MMC_CS_TLS_ENABLE = std::make_pair("ock.mmc.config_store.tls.enable", false);
constexpr auto OCK_MMC_CS_TLS_CA_PATH = std::make_pair("ock.mmc.config_store.tls.ca.path", "");
constexpr auto OCK_MMC_CS_TLS_CRL_PATH = std::make_pair("ock.mmc.config_store.tls.ca.crl.path", "");
constexpr auto OCK_MMC_CS_TLS_CERT_PATH = std::make_pair("ock.mmc.config_store.tls.cert.path", "");
constexpr auto OCK_MMC_CS_TLS_KEY_PATH = std::make_pair("ock.mmc.config_store.tls.key.path", "");
constexpr auto OCK_MMC_CS_TLS_KEY_PASS_PATH = std::make_pair("ock.mmc.config_store.tls.key.pass.path", "");
constexpr auto OCK_MMC_CS_TLS_PACKAGE_PATH = std::make_pair("ock.mmc.config_store.tls.package.path", "");
constexpr auto OCK_MMC_CS_TLS_DECRYPTER_PATH = std::make_pair("ock.mmc.config_store.tls.decrypter.path", "");

constexpr auto OCK_MMC_METRICS_TLS_ENABLE = std::make_pair("ock.mmc.metrics.tls.enable", false);
constexpr auto OCK_MMC_METRICS_TLS_CA_PATH = std::make_pair("ock.mmc.metrics.tls.ca.path", "");
constexpr auto OCK_MMC_METRICS_TLS_CRL_PATH = std::make_pair("ock.mmc.metrics.tls.ca.crl.path", "");
constexpr auto OCK_MMC_METRICS_TLS_CERT_PATH = std::make_pair("ock.mmc.metrics.tls.cert.path", "");
constexpr auto OCK_MMC_METRICS_TLS_KEY_PATH = std::make_pair("ock.mmc.metrics.tls.key.path", "");
constexpr auto OCK_MMC_METRICS_TLS_KEY_PASS_PATH = std::make_pair("ock.mmc.metrics.tls.key.pass.path", "");
constexpr auto OCK_MMC_METRICS_TLS_PACKAGE_PATH = std::make_pair("ock.mmc.metrics.tls.package.path", "");
constexpr auto OCK_MMC_METRICS_TLS_DECRYPTER_PATH = std::make_pair("ock.mmc.metrics.tls.decrypter.path", "");

constexpr auto OKC_MMC_LOCAL_SERVICE_WORLD_SIZE = std::make_pair("ock.mmc.local_service.world_size", 256);
constexpr auto OKC_MMC_LOCAL_SERVICE_BM_IP_PORT =
    std::make_pair("ock.mmc.local_service.config_store_url", "tcp://127.0.0.1:6000");
constexpr auto OKC_MMC_LOCAL_SERVICE_BM_IP_PORT_DOMAIN =
    std::make_pair("ock.mmc.local_service.config_store_url_domain", "");

constexpr auto OKC_MMC_LOCAL_SERVICE_PROTOCOL = std::make_pair("ock.mmc.local_service.protocol", "host_rdma");
constexpr auto OKC_MMC_LOCAL_SERVICE_TAG = std::make_pair("ock.mmc.local_service.tag", "");
constexpr auto OKC_MMC_LOCAL_SERVICE_TAG_OP_INFO = std::make_pair("ock.mmc.local_service.tag_op_info", "");
constexpr auto OKC_MMC_LOCAL_SERVICE_DRAM_SIZE = std::make_pair("ock.mmc.local_service.dram.size", "128MB");
constexpr auto OKC_MMC_LOCAL_SERVICE_MAX_DRAM_SIZE = std::make_pair("ock.mmc.local_service.max.dram.size", "1TB");
constexpr auto OKC_MMC_LOCAL_SERVICE_HBM_SIZE = std::make_pair("ock.mmc.local_service.hbm.size", "0");
constexpr auto OKC_MMC_LOCAL_SERVICE_MAX_HBM_SIZE = std::make_pair("ock.mmc.local_service.max.hbm.size", "0");
constexpr auto OKC_MMC_LOCAL_SERVICE_BM_HCOM_URL =
    std::make_pair("ock.mmc.local_service.hcom_url", "tcp://127.0.0.1:7000");
constexpr auto OCK_MMC_HCOM_TLS_ENABLE = std::make_pair("ock.mmc.local_service.hcom.tls.enable", false);
constexpr auto OCK_MMC_HCOM_TLS_CA_PATH = std::make_pair("ock.mmc.local_service.hcom.tls.ca.path", "");
constexpr auto OCK_MMC_HCOM_TLS_CRL_PATH = std::make_pair("ock.mmc.local_service.hcom.tls.ca.crl.path", "");
constexpr auto OCK_MMC_HCOM_TLS_CERT_PATH = std::make_pair("ock.mmc.local_service.hcom.tls.cert.path", "");
constexpr auto OCK_MMC_HCOM_TLS_KEY_PATH = std::make_pair("ock.mmc.local_service.hcom.tls.key.path", "");
constexpr auto OCK_MMC_HCOM_TLS_KEY_PASS_PATH = std::make_pair("ock.mmc.local_service.hcom.tls.key.pass.path", "");
constexpr auto OCK_MMC_HCOM_TLS_DECRYPTER_PATH = std::make_pair("ock.mmc.local_service.hcom.tls.decrypter.path", "");

constexpr auto OKC_MMC_CLIENT_RETRY_MILLISECONDS = std::make_pair("ock.mmc.client.retry_milliseconds", 0);
constexpr auto OCK_MMC_CLIENT_TIMEOUT_SECONDS = std::make_pair("ock.mmc.client.timeout.seconds", 60);
constexpr auto OCK_MMC_CLIENT_READ_THREAD_POOL_SIZE = std::make_pair("ock.mmc.client.read_thread_pool.size", 4);
constexpr auto OCK_MMC_CLIENT_AGGREGATE_IO = std::make_pair("ock.mmc.client.aggregate.io", true);
constexpr auto OCK_MMC_CLIENT_AGGREGATE_NUM = std::make_pair("ock.mmc.client.aggregate.num", 122);
constexpr auto OCK_MMC_CLIENT_WRITE_THREAD_POOL_SIZE = std::make_pair("ock.mmc.client.write_thread_pool.size", 4);
constexpr auto OCK_MMC_CLIENT_BATCH_CHUNK_SIZE = std::make_pair("ock.mmc.client.batch_option.chunk.size", "8MB");
constexpr auto OCK_MMC_CLIENT_BATCH_CHUNK_COUNT = std::make_pair("ock.mmc.client.batch_option.chunk.count", 3);

constexpr auto OCK_MMC_LOCAL_SERVICE_SSD_SIZE = std::make_pair("ock.mmc.local_service.storage.size", "0");

// KV cache event publisher (opt-in, default disabled).
constexpr auto OCK_MMC_KV_EVENTS_ENABLE = std::make_pair("ock.mmc.kv_events.enable", false);
constexpr auto OCK_MMC_KV_EVENTS_TRANSPORT = std::make_pair("ock.mmc.kv_events.transport", 0);
constexpr auto OCK_MMC_KV_EVENTS_ENDPOINT = std::make_pair("ock.mmc.kv_events.endpoint", "");
constexpr auto OCK_MMC_KV_EVENTS_MODEL_NAME = std::make_pair("ock.mmc.kv_events.model_name", "");
constexpr auto OCK_MMC_KV_EVENTS_TENANT_ID = std::make_pair("ock.mmc.kv_events.tenant_id", "default");
constexpr auto OCK_MMC_KV_EVENTS_BLOCK_SIZE = std::make_pair("ock.mmc.kv_events.block_size", 0);
constexpr auto OCK_MMC_KV_EVENTS_QUEUE_CAPACITY = std::make_pair("ock.mmc.kv_events.queue_capacity", 65536);
constexpr auto OCK_MMC_KV_EVENTS_HASH_AS_INT = std::make_pair("ock.mmc.kv_events.hash_as_int", true);

// ZMQ transport plugin config (independent of kv_events).
constexpr size_t CURVE_KEY_Z85_LEN = 40;
constexpr auto OCK_MMC_ZMQ_ENABLE_CURVE = std::make_pair("ock.mmc.zmq.enable_curve", false);
constexpr auto OCK_MMC_ZMQ_PUBLIC_KEY = std::make_pair("ock.mmc.zmq.public_key", "");
constexpr auto OCK_MMC_ZMQ_SECRET_KEY = std::make_pair("ock.mmc.zmq.private_key", "");
constexpr auto OCK_MMC_PREFETCH_ENABLED = std::make_pair("ock.mmc.storage.prefetch.enabled", false);
constexpr auto OCK_MMC_PENDING_WAIT_TIMEOUT_MS = std::make_pair("ock.mmc.storage.pending_wait.timeout", 300U);
constexpr auto OCK_MMC_LOCAL_SERVICE_STORAGE_ENABLED = std::make_pair("ock.mmc.local_service.storage.enabled", false);
constexpr auto OCK_MMC_ASYNC_FLUSH_INTERVAL_MS = std::make_pair("ock.mmc.storage.async_flush.interval_ms", 200);
constexpr auto OCK_MMC_ASYNC_FLUSH_BATCH_LIMIT = std::make_pair("ock.mmc.storage.async_flush.batch_limit", 20);
// Opt-in ordinary 2MiB DRAM through MF1.3 HYBM_FLAG_UNRESTRICTED_MEM.
constexpr auto OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED =
    std::make_pair("ock.mmc.local_service.dram.unrestricted.enabled", false);

constexpr auto OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT =
    std::make_pair("ock.mmc.local_service.dram.best_effort.enabled", false);

constexpr auto OCK_MMC_DYNAMIC_CONFIG_ENABLE = std::make_pair("ock.mmc.dynamic_config.enable", true);
constexpr auto OCK_MMC_DYNAMIC_CONFIG_INTERVAL =
    std::make_pair("ock.mmc.dynamic_config.interval", DEFAULT_DYNAMIC_CONFIG_INTERVAL);
} // namespace ConfConstant

constexpr int MIN_LOG_ROTATION_FILE_SIZE = 1;
constexpr int MAX_LOG_ROTATION_FILE_SIZE = 500;

constexpr int MIN_LOG_ROTATION_FILE_COUNT = 1;
constexpr int MAX_LOG_ROTATION_FILE_COUNT = 50;

constexpr int MIN_DEVICE_ID = 0;
constexpr int MAX_DEVICE_ID = 383;

constexpr int MIN_WORLD_SIZE = 1;
constexpr int MAX_WORLD_SIZE = 1024;

constexpr int MIN_EVICT_THRESHOLD = 1;
constexpr int MAX_EVICT_THRESHOLD = 99;
constexpr int MIN_LEASE_TTL_MS = 1;
constexpr int MAX_LEASE_TTL_MS = 2147483647;

// GC eviction yield parameter ranges
constexpr int MIN_EVICT_YIELD_INTERVAL = 0;
constexpr int MAX_EVICT_YIELD_INTERVAL = 100000;

constexpr int MIN_EVICT_YIELD_DURATION_US = 0;
constexpr int MAX_EVICT_YIELD_DURATION_US = 10000;

constexpr int MIN_RETRY_MS = 0;
constexpr int MAX_RETRY_MS = 600000;

constexpr int MIN_TIMEOUT_SEC = 1;
constexpr int MAX_TIMEOUT_SEC = 600;

constexpr int MIN_THREAD_POOL_SIZE = 1;
constexpr int MAX_THREAD_POOL_SIZE = 64;
constexpr int MAX_AGGREGATE_NUM = 131072; // 128K
constexpr int MAX_BATCH_CHUNK_COUNT = 64;

constexpr int MIN_DYNAMIC_CONFIG_INTERVAL = 1;
constexpr int MAX_DYNAMIC_CONFIG_INTERVAL = 300;

constexpr uint64_t MAX_DRAM_SIZE = 1024ULL * 1024ULL * 1024ULL * 1024ULL; // 1TB
constexpr uint64_t MAX_HBM_SIZE = 1024ULL * 1024ULL * 1024ULL * 1024ULL;  // 1TB

constexpr uint64_t KB_MEM_BYTES = 1024ULL;
constexpr uint64_t MB_MEM_BYTES = 1024ULL * 1024ULL;
constexpr uint64_t GB_MEM_BYTES = 1024ULL * 1024ULL * 1024ULL;
constexpr uint64_t TB_MEM_BYTES = 1024ULL * 1024ULL * 1024ULL * 1024ULL;

constexpr int MB_NUM = 1024 * 1024;
constexpr uint64_t MEM_2MB_BYTES = 2ULL * 1024ULL * 1024ULL;
constexpr uint64_t MEM_128MB_BYTES = 128ULL * 1024ULL * 1024ULL;

constexpr unsigned long PATH_MAX_LEN = 1023;

constexpr uint64_t DEFAULT_BATCH_CHUNK_SIZE = 8 * MB_MEM_BYTES;

constexpr uint64_t MIN_INTERVAL_SECONDS = 0;
constexpr uint64_t MAX_INTERVAL_SECONDS = 86400;

constexpr int MIN_FLUSH_INTERVAL_MS = 200;
constexpr int MAX_FLUSH_INTERVAL_MS = 1000;
constexpr int MIN_FLUSH_BATCH_LIMIT = 8;
constexpr int MAX_FLUSH_BATCH_LIMIT = 1024;

constexpr int MIN_PENDING_WAIT_TIMEOUT_MS = 1;
constexpr int MAX_PENDING_WAIT_TIMEOUT_MS = 60000;

} // namespace mmc
} // namespace ock

#endif
