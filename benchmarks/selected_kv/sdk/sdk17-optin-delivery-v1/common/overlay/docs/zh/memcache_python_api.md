# MemCache Python API 文档

## 概述

MemCache Python API 是一个高性能的分布式内存缓存客户端，提供了丰富的接口用于数据的存储、检索和管理。该API基于C++核心实现，通过Python绑定提供了简洁易用的接口，支持多种数据传输协议和内存管理模式，适用于大规模分布式计算环境。

### 主要特性

- 支持多种传输协议（TCP、RDMA、SDMA等）。
- 支持主机内存（DRAM）和设备内存（HBM）。
- 提供批量操作接口，提高处理效率。
- 支持数据副本和分层存储。
- 提供灵活的配置选项。
- 支持TLS加密通信。

### 适用场景

- 大规模分布式训练中的数据共享。
- 高性能计算环境中的数据缓存。
- 需要低延迟数据访问的应用场景。
- 跨进程、跨节点的数据交换。

## 安装指南

### 前提条件

- Python >= 3.8 环境（支持源码编译）
- 支持的操作系统：Linux

### 安装步骤

从PyPI安装memcache_hybrid包，该包包含了完整的Python绑定。

```bash
pip install memcache_hybrid
```

📦 包详情：https://pypi.org/project/memcache_hybrid/

## 核心接口

### 接口概述

Python接口封装了MemCache的客户端功能，使用前需要配置并启动独立的meta service进程。

### MetaConfig 类

MetaConfig是元数据服务的启动配置类，用于设置MetaService的监听地址、日志参数、HA开关以及TLS相关参数。

**功能**

创建和管理元数据服务启动配置。

**参数说明**

| 参数名称 | 类型 | 默认值 | 说明 |
| --------- | ------ | ------- | ------ |
| meta_service_url | str | "tcp://127.0.0.1:5000" | 元数据服务启动地址 |
| config_store_url | str | "tcp://127.0.0.1:6000" | Config store地址 |
| metrics_url | str | "http://127.0.0.1:8000" | Metrics HTTP服务地址 |
| ha_enable | bool | false | 是否启用高可用部署 |
| log_level | str | "info" | 日志级别：debug, info, warn, error |
| log_path | str | "/var/log/memcache_hybrid" | 日志目录 |
| log_rotation_file_size | int | 20 | 日志滚动文件大小，单位MB |
| log_rotation_file_count | int | 50 | 日志滚动文件数量 |
| log_output_target | str | "file" | 日志输出目标：screen（仅stdout），file（仅滚动文件），both（stdout + 滚动文件） |
| evict_threshold_high | int | 90 | 驱逐高水位百分比 |
| evict_threshold_low | int | 80 | 驱逐低水位百分比 |
| evict_check_interval_seconds | int | 5 | 周期驱逐检查间隔，单位秒；设置为0时禁用周期检查；HA模式下仅选举出的Master执行周期检查，Standby和角色未知节点不执行 |
| ubs_io_enable | bool | false | 是否启用UBS_IO |
| tls_enable | bool | false | 是否为metaservice启用TLS |
| tls_ca_path | str | "" | TLS CA证书路径 |
| tls_ca_crl_path | str | "" | TLS CA CRL路径 |
| tls_cert_path | str | "" | TLS证书路径 |
| tls_key_path | str | "" | TLS私钥路径 |
| tls_key_pass_path | str | "" | TLS私钥密码路径 |
| tls_package_path | str | "" | TLS包路径 |
| tls_decrypter_path | str | "" | TLS解密库路径 |
| config_store_tls_enable | bool | false | 是否为config_store启用TLS |
| config_store_tls_ca_path | str | "" | Config store TLS CA证书路径 |
| config_store_tls_ca_crl_path | str | "" | Config store TLS CA CRL路径 |
| config_store_tls_cert_path | str | "" | Config store TLS证书路径 |
| config_store_tls_key_path | str | "" | Config store TLS私钥路径 |
| config_store_tls_key_pass_path | str | "" | Config store TLS私钥密码路径 |
| config_store_tls_package_path | str | "" | Config store TLS包路径 |
| config_store_tls_decrypter_path | str | "" | Config store TLS解密库路径 |

**示例**

```python
from memcache_hybrid import MetaConfig

# 创建元数据服务配置对象
config = MetaConfig()
config.meta_service_url = "tcp://127.0.0.1:5000"
config.config_store_url = "tcp://127.0.0.1:6000"
config.metrics_url = "http://127.0.0.1:8000"
config.ha_enable = False
config.log_level = "info"
config.log_path = "/tmp/memcache_hybrid"
config.log_rotation_file_size = 20
config.log_rotation_file_count = 10
config.log_output_target = "both"
config.evict_threshold_high = 90
config.evict_threshold_low = 80
config.evict_check_interval_seconds = 5
```

### MetaService 类

MetaService是MemCache的元数据服务类，用于在Python进程中直接启动元数据服务。通常先通过MetaConfig构造配置，再调用MetaService.setup(config)设置启动参数，最后调用MetaService.main()阻塞启动服务。

#### setup

```python
MetaService.setup(config)
```

**功能**

设置当前进程中MetaService.main()使用的启动配置。

**参数**

`config`：MetaConfig对象，包含元数据服务的启动配置

**返回值**

- `0`：成功
- 其他：失败

**使用示例**

```python
from memcache_hybrid import MetaConfig, MetaService

config = MetaConfig()
config.meta_service_url = "tcp://127.0.0.1:5000"
config.config_store_url = "tcp://127.0.0.1:6000"
config.metrics_url = "http://127.0.0.1:8000"
config.log_level = "info"

MetaService.setup(config)
```

#### main

```python
MetaService.main()
```

**功能**

直接启动元数据服务进程。这是一个阻塞调用，直到服务收到退出信号后才会返回。

**参数**

无

**返回值**

无返回值。服务正常运行时会一直阻塞，服务异常退出时会抛出异常。

**异常处理**

- 服务启动失败时会抛出RuntimeError异常，包含具体错误信息。
- 服务运行过程中遇到致命错误时会抛出RuntimeError异常。
- 服务被信号中断时会抛出KeyboardInterrupt异常。

**使用示例**

```python
from memcache_hybrid import MetaConfig, MetaService

config = MetaConfig()
config.meta_service_url = "tcp://127.0.0.1:5000"
config.config_store_url = "tcp://127.0.0.1:6000"
config.metrics_url = "http://127.0.0.1:8000"
config.log_level = "info"

try:
    assert MetaService.setup(config) == 0, "setup meta config failed"
    MetaService.main()
except KeyboardInterrupt:
    print("MetaService stopped by user")
except RuntimeError as e:
    print(f"MetaService failed to start or encountered an error: {e}")
```

**注意事项**

- `MetaService.main()` 是阻塞调用，建议在单独的进程中运行。
- 如果未调用 `MetaService.setup(config)`，服务会尝试读取环境变量 `MMC_META_CONFIG_PATH` 指向的配置文件。
- 服务停止后，所有依赖该服务的客户端连接将失效。

### LocalConfig 类

LocalConfig是分布式内存缓存的配置类，用于设置客户端的各种参数。

**功能**

创建和管理客户端配置。

**参数说明**

| 参数名称 | 类型 | 默认值 | 说明 |
| --------- | ------ | ------- | ------ |
| meta_service_url | str | "tcp://127.0.0.1:5000" | Meta service的URL地址，格式如"tcp://192.168.1.1:5000" |
| config_store_url | str | "tcp://127.0.0.1:6000" | Config store的URL地址 |
| log_level | str | "info" | 日志级别：debug, info, warn, error |
| world_size | int | 256 | 最大支持的rank数量 |
| protocol | str | "host_rdma" | 数据传输协议：host_rdma, host_urma, host_tcp, host_shm, device_rdma, device_urma, device_uboe, device_sdma |
| hcom_url | str | "tcp://127.0.0.1:7000" | HCOM URL for RDMA network |
| dram_size | str | "1GB" | DRAM空间使用，支持格式如1GB，2MB等 |
| hbm_size | str | "0" | HBM空间使用 |
| max_dram_size | str | "1TB" | 所有本地进程中dram_size的最大值 |
| max_hbm_size | str | "0" | 所有本地进程中hbm_size的最大值 |
| client_retry_milliseconds | int | 0 | 客户端请求meta service时的总重试时间（毫秒） |
| client_timeout_seconds | int | 60 | 客户端请求超时时间(秒) |
| read_thread_pool_size | int | 4 | 读线程池大小 |
| write_thread_pool_size | int | 4 | 写线程池大小 |
| aggregate_io | bool | true | 是否启用读/写聚合 |
| aggregate_num | int | 122 | 聚合数量 |
| ubs_io_enable | bool | false | 是否启用UBS_IO |
| tls_enable | bool | false | 是否为metaservice启用TLS |
| tls_ca_path | str | "" | TLS CA证书路径 |
| tls_ca_crl_path | str | "" | TLS CA CRL路径 |
| tls_cert_path | str | "" | TLS证书路径 |
| tls_key_path | str | "" | TLS密钥路径 |
| tls_key_pass_path | str | "" | TLS密钥密码路径 |
| tls_package_path | str | "" | TLS包路径 |
| tls_decrypter_path | str | "" | TLS解密库路径 |
| config_store_tls_enable | bool | false | 是否为config_store启用TLS |
| config_store_tls_ca_path | str | "" | Config store TLS CA证书路径 |
| config_store_tls_ca_crl_path | str | "" | Config store TLS CA CRL路径 |
| config_store_tls_cert_path | str | "" | Config store TLS证书路径 |
| config_store_tls_key_path | str | "" | Config store TLS密钥路径 |
| config_store_tls_key_pass_path | str | "" | Config store TLS密钥密码路径 |
| config_store_tls_package_path | str | "" | Config store TLS包路径 |
| config_store_tls_decrypter_path | str | "" | Config store TLS解密库路径 |
| hcom_tls_enable | bool | false | 是否为HCOM启用TLS |
| hcom_tls_ca_path | str | "" | HCOM TLS CA证书路径 |
| hcom_tls_ca_crl_path | str | "" | HCOM TLS CA CRL路径 |
| hcom_tls_cert_path | str | "" | HCOM TLS证书路径 |
| hcom_tls_key_path | str | "" | HCOM TLS密钥路径 |
| hcom_tls_key_pass_path | str | "" | HCOM TLS密钥密码路径 |
| hcom_tls_decrypter_path | str | "" | HCOM TLS解密库路径 |

**示例**

```python
# 创建配置对象
config = LocalConfig()
config.meta_service_url = "tcp://192.168.1.1:5000"
config.protocol = "device_sdma"
config.dram_size = "2GB"
config.hbm_size = "8GB"
config.client_timeout_seconds = 30
```

### DistributedObjectStore 类

DistributedObjectStore是分布式对象存储的Python接口封装类，提供了完整的分布式内存缓存操作接口。

#### \_\_init\_\_

```python
store = DistributedObjectStore()
```

**功能**

创建DistributedObjectStore实例。

#### setup

```python
result = store.setup(config)
```

**功能**

设置分布式内存缓存客户端配置。

**参数**

`config`：LocalConfig对象，包含客户端的所有配置参数。

**返回值**

- `0`：成功
- 其他：失败

#### init

```python
result = store.init(device_id, init_bm=True)
```

**功能**

初始化分布式内存缓存客户端。

**参数**

- `device_id`：使用HBM时的NPU卡用户ID（支持ASCEND_RT_VISIBLE_DEVICES映射）。
- `init_bm`：是否初始化BM提供内存，默认值为 True。设 False 时将启动纯client模式，不支持数据读写操作。

**返回值**

- `0`：成功
- 其他：失败

#### get_local_service_id

```python
store.get_local_service_id()
```

**功能**

获取本地服务ID。

#### close

```python
store.close()
```

**功能**

关闭并清理分布式内存缓存客户端。

#### put

```python
result = store.put(key, buf, replicateConfig=defaultConfig)
```

**功能**

将指定key的数据写入分布式内存缓存中。

**参数**

- `key`：数据的键，字符串类型
- `buf`：要存储的字节数据
- `replicateConfig`：复制配置，具体请参考ReplicateConfig数据结构

**返回值**

- `0`：成功
- 其他：失败

#### put_batch

```python
def put_batch(self, keys: List[str], values: List[bytes], replicateConfig: ReplicateConfig = defaultConfig) -> int
```

**功能**

在单个批处理操作中存储多个对象，提高处理效率。

**参数**

- `keys` (List[str])：对象标识符列表。
- `values` (List[bytes])：要存储的二进制数据列表。
- `replicateConfig` (ReplicateConfig, optional)：所有对象的复制配置。

**返回值**

`int`：状态码 (0：成功；非零：失败错误码)

**示例**

```python
keys = ["key1", "key2", "key3"]
values = [b"value1", b"value2", b"value3"]
result = store.put_batch(keys, values)
assert result == 0, f"put_batch failed: {result}"
```

#### put_from

```python
result = store.put_from(key, buffer_ptr, size, direct=memcache_hybrid.H2G, replicateConfig=defaultConfig)
```

**功能**

从预分配的缓冲区中写入数据，适用于需要高效内存管理的场景。

**参数**

- `key`：数据的键。
- `buffer_ptr`：缓冲区指针。
- `size`：数据大小。
- `direct`：数据拷贝方向，可选值：
    - `memcache_hybrid.H2G`：从主机内存到全局内存（默认）。
    - `memcache_hybrid.L2G`：从卡上内存到全局内存。
- `replicateConfig`：复制配置，具体请参考ReplicateConfig数据结构。

**返回值**

- `0`：成功
- 其他：失败

#### batch_put_from

```python
result = store.batch_put_from(keys, buffer_ptrs, sizes, direct=memcache_hybrid.H2G, replicateConfig=defaultConfig)
```

**功能**

从预分配的缓冲区中批量写入数据，提高处理效率。

**参数**

- `keys`：键列表。
- `buffer_ptrs`：缓冲区指针列表。
- `sizes`：数据大小列表。
- `direct`：数据拷贝方向，可选值：
    - `memcache_hybrid.H2G`：从主机内存到全局内存（默认）。
    - `memcache_hybrid.L2G`：从卡上内存到全局内存。
- `replicateConfig`：复制配置，具体请参考ReplicateConfig数据结构。

**返回值**

结果列表，每个元素表示对应写入操作的结果：

- `0`：成功
- 其他：错误

#### put_from_layers

```python
result = store.put_from_layers(key, buffer_ptrs, sizes, direct=memcache_hybrid.H2G, replicateConfig=defaultConfig)
```

**功能**

从多个预分配的缓冲区中写入分层数据，适用于复杂数据结构的存储。

**参数**

- `key`：数据的键。
- `buffer_ptrs`：缓冲区指针列表，每个指针对应一个数据层。
- `sizes`：数据大小列表，每个元素对应一个数据层的大小。
- `direct`：数据拷贝方向，可选值：
    - `memcache_hybrid.H2G`：从主机内存到全局内存（默认）。
    - `memcache_hybrid.L2G`：从卡上内存到全局内存。
- `replicateConfig`：复制配置，具体请参考ReplicateConfig数据结构。

**返回值**

- `0`：成功
- 其他：失败

#### batch_put_from_layers

```python
result = store.batch_put_from_layers(keys, buffer_ptrs_list, sizes_list, direct=memcache_hybrid.H2G,
                                     replicateConfig=defaultConfig)
```

**功能**

从多个预分配的缓冲区中批量写入分层数据，提高处理效率。

**参数**

- `keys`：数据键列表，每个键对应一个分层数据对象。
- `buffer_ptrs_list`：缓冲区指针二维列表，外层列表对应每个键，内层列表对应每个键的各个数据层指针。
- `sizes_list`：数据大小二维列表，外层列表对应每个键，内层列表对应每个键的各个数据层大小。
- `direct`：数据拷贝方向，可选值：
    - `memcache_hybrid.H2G`：从主机内存到全局内存（默认）。
    - `memcache_hybrid.L2G`：从卡上内存到全局内存。
- `replicateConfig`：复制配置，具体请参考ReplicateConfig数据结构。

**返回值**

结果列表，每个元素表示对应写入操作的结果：

  - `0`：成功
  - 其他：错误

#### get

```python
data = store.get(key)
```

**功能**

获取指定key的数据。

**参数**

`key`：数据的键。

**返回值**

- 成功时返回数据字节串。
- 失败时返回空字节串。

#### get_batch

```python
def get_batch(self, keys: List[str]) -> List[bytes]
```

**功能**

在单个批处理操作中检索多个对象，提高处理效率。

**参数**

`keys`(List[str])：要检索的对象标识符列表。

**返回值**

- `List[bytes]`：检索到的二进制数据列表。
- 成功时返回二进制数据列表，失败时返回空二进制数据列表。

**示例**

```python
keys = ["key1", "key2", "key3"]
values = store.get_batch(keys)
for key, value in zip(keys, values):
    print(f"{key}: {len(value)} bytes")
```

#### get_into

```python
result = store.get_into(key, buffer_ptr, size, direct=memcache_hybrid.G2H)
```

**功能**

将数据直接获取到预分配的缓冲区中，适用于需要高效内存管理的场景。

**参数**

- `key`：数据的键。
- `buffer_ptr`：目标缓冲区指针。
- `size`：缓冲区大小。
- `direct`：数据拷贝方向，可选值：
    - `memcache_hybrid.G2H`：从全局内存到主机内存（默认）。
    - `memcache_hybrid.G2L`：从全局内存到卡上内存。

**返回值**

- `0`：成功
- 其他：失败

#### batch_get_into

```python
results = store.batch_get_into(keys, buffer_ptrs, sizes, direct=memcache_hybrid.G2H)
```

**功能**

批量将数据获取到预分配的缓冲区中，提高处理效率。

**参数**

- `keys`：键列表。
- `buffer_ptrs`：缓冲区指针列表。
- `sizes`：缓冲区大小列表。
- `direct`：数据拷贝方向，可选值：
    - `memcache_hybrid.G2H`：从全局内存到主机内存（默认）。
    - `memcache_hybrid.G2L`：从全局内存到卡上内存。

**返回值**

结果列表，每个元素表示对应操作的结果：

- `0`：成功
- 其他：错误

#### get_into_layers

```python
result = store.get_into_layers(key, buffer_ptrs, sizes, direct=memcache_hybrid.G2H)
```

**功能**

将数据分层获取到预分配的缓冲区中，适用于复杂数据结构的读取。

**参数**

- `key`：数据的键。
- `buffer_ptrs`：目标缓冲区指针列表，每个指针对应一个数据层。
- `sizes`：缓冲区大小列表，每个元素对应一个数据层的大小。
- `direct`：数据拷贝方向，可选值：
    - `memcache_hybrid.G2H`：从全局内存到主机内存（默认）。
    - `memcache_hybrid.G2L`：从全局内存到卡上内存。

**返回值**

- `0`：成功
- 其他：失败

#### batch_get_into_layers

```python
results = store.batch_get_into_layers(keys, buffer_ptrs_list, sizes_list, direct=memcache_hybrid.G2H)
```

**功能**

批量将分层数据获取到预分配的缓冲区中，提高处理效率。

**参数**

- `keys`：数据键列表，每个键对应一个分层数据对象。
- `buffer_ptrs_list`：缓冲区指针二维列表，外层列表对应每个键，内层列表对应每个键的各个目标数据层指针。
- `sizes_list`：缓冲区大小二维列表，外层列表对应每个键，内层列表对应每个键的各个数据层大小。
- `direct`：数据拷贝方向，可选值：
    - `memcache_hybrid.G2H`：从全局内存到主机内存（默认）。
    - `memcache_hybrid.G2L`：从全局内存到卡上内存。

**返回值**

结果列表，每个元素表示对应操作的结果：

- `0`：成功
- 其他：错误

#### remove

```python
result = store.remove(key)
```

**功能**

删除指定key的数据。

**参数**

`key`：要删除的数据键。

**返回值**

- `0`：成功
- 其他：失败

#### remove_batch

```python
results = store.remove_batch(keys)
```

**功能**

批量删除数据，提高处理效率。

**参数**

`keys`：要删除的键列表。

**返回值**

结果列表，每个元素表示对应删除操作的结果：

- `0`：成功
- 其他：错误

#### remove_all

```python
result = store.remove_all()
```

**功能**

删除内存池中所有的键值对。

**返回值**

- `0`：成功
- 其他：失败

#### is_exist

```python
result = store.is_exist(key)
```

**功能**

检查指定key是否存在。

**参数**

`key`：要检查的键。

**返回值**

- `1`：存在
- `0`：不存在
- 其他：错误

#### batch_is_exist

```python
results = store.batch_is_exist(keys)
```

**功能**

批量检查键是否存在，提高处理效率。

**参数**

`keys`：要检查的键列表。

**返回值**

结果列表，每个元素表示对应键的存在状态：

- `1`：存在
- `0`：不存在
- 其他：错误

#### get_key_info

```python
key_info = store.get_key_info(key, flag=0)
```

**功能**

获取指定key的数据信息。

**参数**

- `key`: 数据的键
- `flag`: 查询标志，Python 接口默认值为 `GET_KEY_INFO_FOR_LAYER_WISE`（LayerWise模式查询，无 DRAM 时从 SSD 回温）。

**返回值**

`KeyInfo`对象，包含以下方法：

  - `size()`：获取数据大小。
  - `loc_list()`：获取数据位置列表。
  - `type_list()`：获取数据类型列表。
  - `gva_list()`：获取每个 blob 对应的 GVA 列表。
  - `__str__()`：获取信息的字符串表示。

#### batch_get_key_info

```python
key_infos = store.batch_get_key_info(keys, flag=0)
```

**功能**

批量获取多个key的数据信息，提高处理效率。

**参数**

- `keys`: 数据键列表
- `flag`: 查询标志，Python 接口默认值为 `GET_KEY_INFO_FOR_LAYER_WISE`（LayerWise模式查询，无 DRAM 时从 SSD 回温）。

**返回值**

`KeyInfo`对象列表，每个对象包含以下方法：

  - `size()`：获取数据大小。
  - `loc_list()`：获取数据位置列表。
  - `type_list()`：获取数据类型列表。
  - `gva_list()`：获取每个 blob 对应的 GVA 列表。
  - `__str__()`：获取信息的字符串表示。

#### prefetch

```python
result = store.prefetch(keys, src_media=2, dst_media=1, flags=0)
```

**功能**

将指定 keys 从源介质回温到目标介质。当前仅支持 SSD→DRAM（`src_media=2, dst_media=1`），其他组合暂不支持。

**参数**

- `keys`：要回温的 key 列表。
- `src_media`：源介质类型（0=HBM，1=DRAM，2=SSD），默认 `2`（SSD），预留参数。
- `dst_media`：目标介质类型，默认 `1`（DRAM），预留参数。
- `flags`：预留标志，必须为 `0`。

**返回值**

- `int`：`0` 表示成功，其他值表示失败。

**示例**

```python
result = store.prefetch(["key1", "key2"])
assert result == 0
```

#### evict

```python
result = store.evict(keys, src_media=1, dst_media=2, flags=0)
```

**功能**

将指定 keys 从源介质驱逐到目标介质。当前仅支持 DRAM→SSD（`src_media=1, dst_media=2`），其他组合暂不支持。当 SSD 不可用或 DRAM 使用率高于回温水线时，DRAM 上的数据会被直接删除。

**参数**

- `keys`：要驱逐的 key 列表。
- `src_media`：源介质类型（0=HBM，1=DRAM，2=SSD），默认 `1`（DRAM），预留参数。
- `dst_media`：目标介质类型，默认 `2`（SSD），预留参数。
- `flags`：预留标志，必须为 `0`。

**返回值**

- `int`：`0` 表示成功，其他值表示失败。

**示例**

```python
result = store.evict(["key1", "key2"])
assert result == 0
```

#### GVA 相关接口与常量

#### get_key_info / batch_get_key_info 补充说明

- 这两个接口的实际签名分别为 `store.get_key_info(key, flag=0)` 和 `store.batch_get_key_info(keys, flag=0)`。
- `KeyInfo` 除 `size()`、`loc_list()`、`type_list()` 外，还提供 `gva_list()`，用于返回每个 blob 对应的 GVA 列表。

```python
key_info = store.get_key_info(key)
key_infos = store.batch_get_key_info(keys)
gvas = key_info.gva_list()
```

#### batch_add_lease

```python
results = store.batch_add_lease(keys, leaseTtlMs=0)
```

**功能**

批量为多个 key 增加读租约，并记录后续 GVA 读取所需的读租约状态。

**参数**

- `keys`：要增加读租约的 key 列表，不能为空。
- `leaseTtlMs`：要增加的租约时间，单位为毫秒，默认为 `0`。为 `0` 时使用 meta 侧配置项
  `ock.mmc.meta.lease_ttl_ms`。

**返回值**

`List[int]`：每个元素表示对应 key 的处理结果，`0` 表示成功，其他值表示失败。返回列表长度与 `keys` 一致。

**使用说明**

- 该接口返回每个 key 的错误码，不返回 `KeyInfo`。
- 该接口为非事务接口；某个 key 失败不会回滚其他 key 已经成功增加的读租约。
- `leaseTtlMs` 为 `0` 时，meta 侧使用配置项 `ock.mmc.meta.lease_ttl_ms` 的值增加租约。
- 调用方应只对返回值为 `0` 的 key 继续执行后续基于 GVA 的读取流程。
- 典型用法是先通过 `batch_get_key_info(keys, flag=0)` 获取 GVA，再调用 `batch_add_lease(keys)`
  为后续基于 GVA 的读取流程建立读租约状态。
- 同一进程对同一 key 重复调用时，会复用当前进程中已有的读租约并在 meta 侧续租；
  完成读取后调用一次 `batch_remove_lease` 即可释放。

```python
key_infos = store.batch_get_key_info(keys, flag=0)
gvas = [info.gva_list()[0] for info in key_infos]
results = store.batch_add_lease(keys)
valid_indices = [idx for idx, ret in enumerate(results) if ret == 0]
store.batch_copy(
    [gvas[idx] for idx in valid_indices],
    [buffer_ptrs[idx] for idx in valid_indices],
    [sizes[idx] for idx in valid_indices],
    direct=SMEMB_COPY_G2L,
)
store.batch_remove_lease([keys[idx] for idx in valid_indices])
```

#### batch_remove_lease

```python
ret = store.batch_remove_lease(keys)
```

**功能**

批量移除多个 key 的读租约，并清理当前进程中对应的 GVA 读取状态。

**参数**

`keys`：要移除读租约的 key 列表，不能为空。

**返回值**

- `int`：`0` 表示本地读租约检查通过并已触发移除租约请求发送流程。
- 其他值表示失败。

**使用说明**

调用方完成基于 GVA 的读取后，应调用该接口显式释放由 `batch_add_lease` 建立的读租约。

#### batch_alloc

```python
gvas = store.batch_alloc(keys, sizes, media=1, leaseTtlMs=0)
```

**功能**

批量为多个 key 申请全局内存，并返回每个 key 对应的起始 GVA。

**参数**

- `keys`：要申请内存的 key 列表。
- `sizes`：每个 key 对应的数据大小列表，长度必须与 `keys` 一致。
- `media`：申请的介质类型，默认值为 `1`（`MEDIA_DRAM`）；`0` 表示 `MEDIA_HBM`。
- `leaseTtlMs`：要增加的租约时间，单位为毫秒，默认为 `0`。为 `0` 时使用 meta 侧配置项 `ock.mmc.meta.lease_ttl_ms`。

**返回值**

- `List[int]`：每个元素为一个 key 对应的起始 GVA。
- 申请失败或参数非法时，对应元素为 `0`。

#### batch_copy

```python
result = store.batch_copy(gva_ptrs, buffer_ptrs, sizes, direct=SMEMB_COPY_G2L)
```

**功能**

批量在 GVA 地址与本地缓冲区之间执行数据拷贝。

> [!NOTE] 说明
> 写方向（`SMEMB_COPY_L2G`/`SMEMB_COPY_H2G`）只拷贝数据，**不会**将 GVA 对应的 blob 状态翻转为 READABLE；调用方必须再调用 `batch_write_finish` 显式通知写完成。

**参数**

- `gva_ptrs`：GVA 地址列表。
- `buffer_ptrs`：本地缓冲区指针列表，必须与 `gva_ptrs` 一一对应。
- `sizes`：每次拷贝的大小列表，长度必须与 `gva_ptrs` 一致。
- `direct`：数据拷贝方向，常用值包括：
    - `SMEMB_COPY_L2G`（`0`）：本地 HBM -> 全局内存。
    - `SMEMB_COPY_G2L`（`1`）：全局内存 -> 本地 HBM。
    - `SMEMB_COPY_G2H`（`2`）：全局内存 -> 本地 Host DRAM。
    - `SMEMB_COPY_H2G`（`3`）：本地 Host DRAM -> 全局内存。

**返回值**

- `0`：成功
- 其他：失败

#### batch_copy_layers

```python
result = store.batch_copy_layers(gva_ptrs, buffer_ptrs, sizes, direct=SMEMB_COPY_G2L)
```

**功能**

以分层数据的形式批量在 GVA 地址与本地缓冲区之间执行拷贝。

> [!NOTE] 说明
> 与 `batch_copy` 一致，写方向只拷贝数据，不翻转 blob 状态；写完后需调用 `batch_write_finish`。

**参数**:

- `gva_ptrs`：每个对象的起始 GVA 列表。
- `buffer_ptrs`：二维缓冲区指针列表，外层对应对象，内层对应该对象的各层缓冲区。
- `sizes`：二维大小列表，每层大小会从对应起始 GVA 开始按顺序累加。
- `direct`：数据拷贝方向，含义与 `batch_copy` 一致。

**返回值**

- `0`：成功
- 其他：失败

#### batch_write_finish

```python
results = store.batch_write_finish(keys, res)
```

**功能**

显式通知 meta service 给定 key 的写入已完成。

> [!NOTE] 说明
>
> - 调用方在 `batch_copy` / `batch_copy_layers` 写方向完成后必须调用此接口，meta service 才会将对应 blob 从 `ALLOCATED` 翻转为 `READABLE`，此后其他进程才能通过 `batch_get_key_info` / `batch_copy` 读方向读到数据。
> - 对于已处于 `READABLE` 的 key，再次调用为幂等并直接返回成功。
> - 对未分配过的 key 返回 `MMC_UNMATCHED_KEY`。

**参数**

- `keys`：已写入完成的 key 列表。
- `res`：与 `keys` 等长的每 key 写入结果，`0` 表示成功（对应 `MMC_WRITE_OK`，blob 翻为 `READABLE`），非 `0` 表示失败（对应 `MMC_WRITE_FAIL`，meta service 会移除该 blob）。

**返回值**

- `List[int]`：与 `keys` 等长，每个元素为 meta service 对该 key 的实际更新结果。
- `0` 表示成功。

> [!NOTE] 典型 GVA 跨进程读取流程：
>
> - 写进程：`batch_alloc` -> `batch_copy` / `batch_copy_layers`（写入数据）-> `batch_write_finish` 显式通知写完成，blob 翻为 READABLE。
> - 读进程：`batch_get_key_info(...)` 获取 GVA -> `batch_add_lease(...)` 增加读租约并准备 GVA 读取状态 -> `batch_copy` / `batch_copy_layers`（读取）-> `batch_remove_lease(...)` 显式释放读租约。

#### register_buffer

```python
result = store.register_buffer(buffer_ptr, size)
```

**功能**

注册内存缓冲区，用于Device RDMA的加速操作。

**参数**

- `buffer_ptr`：缓冲区指针
- `size`：缓冲区大小

**返回值**:

- `0`：成功
- 其他：失败

#### unregister_buffer

```python
result = store.unregister_buffer(buffer_ptr, size)
```

**功能**

注销内存缓冲区，用于Device RDMA的加速操作。

**参数**

- `buffer_ptr`：缓冲区指针
- `size`：缓冲区大小

**返回值**

- `0`：成功
- 其他：失败

## 使用示例

### 基本使用示例（推荐）

以下示例展示了如何使用LocalConfig配置并使用DistributedObjectStore：

```python
from memcache_hybrid import LocalConfig, DistributedObjectStore, ReplicateConfig

# 创建配置对象
config = LocalConfig()
config.meta_service_url = "tcp://192.168.1.1:5000"
config.config_store_url = "tcp://192.168.1.1:6000"
config.protocol = "device_sdma"
config.dram_size = "10GB"
config.hbm_size = "5GB"
config.client_timeout_seconds = 30

# 初始化存储客户端
store = DistributedObjectStore()
result = store.setup(config)  # 设置配置
if result != 0:
    raise Exception(f"Failed to setup store: {result}")

result = store.init(device_id=0, init_bm=True)
if result != 0:
    raise Exception(f"Failed to init store: {result}")

# 执行数据操作
# 创建复制配置
replica_cfg = ReplicateConfig()
replica_cfg.replicaNum = 3
replica_cfg.preferredLocalServiceIDs = [101, 102, 103]

# 存储数据
buffer = bytearray(b"example_data")
result = store.put("key1", buffer, replica_cfg)
if result != 0:
    raise Exception(f"Failed to put data: {result}")

# 检索数据
data = store.get("key1")
print(f"Retrieved data: {data}")

# 检查数据是否存在
exists = store.is_exist("key1")
print(f"Data exists: {exists}")

# 删除数据
result = store.remove("key1")
if result != 0:
    raise Exception(f"Failed to remove data: {result}")

# 关闭连接
store.close()
```

### 使用环境变量进行配置

以下示例展示了如何使用环境变量配置客户端（兼容原有方式）：

`export MMC_LOCAL_CONFIG_PATH=/path/to/local_config.conf`

```python
import os
from memcache_hybrid import DistributedObjectStore

# 创建并初始化客户端
store = DistributedObjectStore()
result = store.init(device_id=0)
if result != 0:
    raise Exception(f"Failed to init store: {result}")

# 执行数据操作
buffer = bytearray(b"example_data")
result = store.put("key1", buffer)
if result != 0:
    raise Exception(f"Failed to put data: {result}")

# 检索数据
data = store.get("key1")
print(f"Retrieved data: {data}")

# 关闭连接
store.close()
```

### 批量操作示例

以下示例展示了如何使用批量操作提高处理效率：

```python
from memcache_hybrid import LocalConfig, DistributedObjectStore

# 创建配置和客户端
config = LocalConfig()
config.meta_service_url = "tcp://192.168.1.1:5000"
config.protocol = "device_sdma"

store = DistributedObjectStore()
store.setup(config)
store.init(device_id=0, init_bm=True)

# 批量存储数据
keys = ["key1", "key2", "key3", "key4", "key5"]
values = [b"value1", b"value2", b"value3", b"value4", b"value5"]
result = store.put_batch(keys, values)
if result != 0:
    raise Exception(f"Failed to put batch: {result}")

# 批量检索数据
retrieved_values = store.get_batch(keys)
for key, value in zip(keys, retrieved_values):
    print(f"{key}: {value}")

# 批量检查数据是否存在
existence = store.batch_is_exist(keys)
print(f"Existence: {existence}")

# 批量删除数据
results = store.remove_batch(keys)
print(f"Remove results: {results}")

# 关闭连接
store.close()
```

## 数据结构与枚举类型

### BmCopyType 枚举类型

用于指定数据拷贝方向的枚举类型：

- `H2G`：从主机内存到全局内存。
- `L2G`：从卡上内存到全局内存。
- `G2H`：从全局内存到主机内存。
- `G2L`：从全局内存到卡上内存。
- `AUTO`：根据内存类型自动选择拷贝方向。

### LocalConfig 配置类

客户端配置结构体，用于设置分布式内存缓存客户端的各种参数。详情请参考[LocalConfig 类](#localconfig-类)的参数说明。

### ReplicateConfig 配置类

客户端复制配置结构体，用于设置数据复制策略：

- `replicaNum`：副本数，最大为8，默认为1。
- `preferredLocalServiceIDs`：强制分配的实例ID列表，列表大小必须小于或等于replicaNum。

## 错误码

| 错误码   | 说明        |
|-------|-----------|
| 0     | 操作成功      |
| -1    | 一般错误      |
| -3000 | 参数无效      |
| -3001 | 内存分配失败    |
| -3002 | 对象创建失败    |
| -3003 | 服务未启动     |
| -3004 | 操作超时      |
| -3005 | 重复调用      |
| -3006 | 对象已存在     |
| -3007 | 对象不存在     |
| -3008 | 未初始化      |
| -3009 | 网络序列号重复   |
| -3010 | 网络序列号未找到  |
| -3011 | 已通知       |
| -3012 | 内部错误      |
| -3013 | 超出容量限制    |
| -3014 | 连接未找到     |
| -3015 | 网络请求句柄未找到 |
| -3016 | 内存不足      |
| -3017 | 未连接到元数据服务 |
| -3018 | 未连接到本地服务  |
| -3019 | 客户端未初始化   |
| -3101 | 状态不匹配     |
| -3102 | 键不匹配      |
| -3103 | 返回值不匹配    |
| -3104 | 租约未到期     |
| -3105 | 元数据备份失败 |

## 注意事项

- 所有键的长度必须小于256字节。
- 支持同步和异步两种操作模式。
- 批量操作可以显著提高处理效率，建议在大量数据操作时使用。
- TLS配置需要确保证书路径和权限正确。
- 客户端需要与meta service保持网络连通性。
- 初始化时设置init_bm为False将启动纯client模式，不支持数据读写操作。


### 可选扩展：`batch_add_lease_with_info`

```python
statuses, infos = store.batch_add_lease_with_info(keys, lease_ttl_ms=0)
```

两列均与输入 key 同序同长，`infos` 使用现有 `KeyInfo` 类型。每批只调用一次 Add RPC，
不再发 Query 或 Prefetch；仅在真实 Add 及本地 lease tracker 登记成功后发布对应 KeyInfo。
失败项返回非零状态和空 KeyInfo。空输入返回 `([], [])`，不初始化 SDK、不取得租约。
调用者须在调用前记录 pending ownership；异常、丢响应或长度不匹配不能重试 Add。
对真实成功项沿现有 RemoveLease 释放；未知项保留 ownership 并进入原异常清理。
能力检测必须在任何 Add 前完成；不支持该扩展时继续旧 Add 加 Get 流程。
此接口不延长跨 step 的地址生命周期，也不改变 Native 的原有 API。
