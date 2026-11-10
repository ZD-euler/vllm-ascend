# MemCache API

## C++接口列表

C++语言接口功能齐全，基于面向对象设计，提供统一的 `ObjectStore` 抽象基类，封装了实例管理、缓冲区注册、数据操作、批量处理及分层张量等功能。

### 1. 实例创建与生命周期管理接口

#### local_config

`local_config` 是 `Setup` 接口使用的本地配置类型，定义位于 `src/memcache/include/mmc.h`。

> [!NOTE] 说明
> 建议先通过 `create_default_local_config()` 获取带默认值的配置对象，再按需覆盖字段。

**常用字段**

- `meta_service_url`：元服务地址。
- `config_store_url`：配置存储服务地址。
- `log_level`：日志级别，如 `debug`、`info`、`warn`、`error`。
- `world_size`：最大 rank 数。
- `protocol`：数据传输协议，如 `host_rdma`、`host_urma`、`host_tcp`、`device_rdma`、`device_urma`、`device_uboe`、`device_sdma`。
- `hcom_url`：HCOM 服务地址。
- `dram_size` / `hbm_size`：本地服务 DRAM / HBM 容量。
- `max_dram_size` / `max_hbm_size`：所有本地进程可使用的 DRAM / HBM 总上限。

#### create_default_local_config

```c++
local_config create_default_local_config();
```

**功能**

创建一个带内置默认值的 `local_config` 对象，便于用户只覆盖必要配置项。

**返回值**

返回默认初始化后的 `local_config`。

#### ObjectStore::CreateObjectStore

```c++
static std::shared_ptr<ObjectStore> CreateObjectStore();
```

**功能**

创建一个分布式内存缓存存储实例。

**返回值**

返回 std::shared_ptr 管理的智能指针，确保资源自动释放，非空 shared_ptr 表示成功。

#### Setup

```c++
virtual int Setup(const local_config &config);
```

**功能**

初始化并校验本地配置，供后续 `Init` 使用。

**参数**

`config`：本地配置（`local_config`）。

**返回值**

- `0`：成功
- 其他：失败

**推荐调用顺序**

```c++
auto store = ock::mmc::ObjectStore::CreateObjectStore();
local_config config = create_default_local_config();

// 可按需覆盖默认配置，或仅设置 config_path 指向配置文件
int ret = store->Setup(config);
if (ret != 0) {
    return ret;
}

ret = store->Init(0, true);
if (ret != 0) {
    return ret;
}
```

#### Init

```c++
virtual int Init(const uint32_t deviceId, bool initBm = true) = 0;
```

**功能**

初始化当前存储实例，绑定到指定设备。

**参数**

- `deviceId`：目标设备ID。
- `initBm`：是否初始化BM提供内存，默认值为 true。设 false 时将启动纯client模式，不支持数据读写操作。

**返回值**

- `0`：成功
- 其他：失败

#### TearDown

```c++
virtual int TearDown() = 0;
```

**功能**

释放当前实例占用的所有资源，断开与元服务和本地服务的连接。

**返回值**

- `0`：成功
- 其他：失败

### 2. 缓冲区注册接口

#### RegisterBuffer

```c++
virtual int RegisterBuffer(void *buffer, size_t size) = 0;
```

**功能**

将用户分配的内存区域注册到系统中，以启用 RDMA 或零拷贝传输。

**参数**

- `buffer`：内存起始地址。
- `size`：缓冲区字节大小。

**返回值**

- `0`：成功
- 其他：失败

#### UnRegisterBuffer

```c++
virtual int UnRegisterBuffer(void *buffer, size_t size) = 0;
```

**功能**

注销已注册的内存区域。

**参数**

- `buffer`：内存起始地址。
- `size`：缓冲区字节大小。

**返回值**

- `0`：成功
- 其他：失败

### 3. 数据操作接口

#### GetInto

```c++
virtual int GetInto(const std::string &key, void *buffer, size_t size, const int32_t direct = 2) = 0;
```

**功能**

将指定键的数据读入预分配的 buffer 中。

**参数**

- `key`：数据键（长度 < 256字节）。
- `buffer`：目标内存地址。
- `size`：缓冲区容量。
- `direct`：数据流向。

**返回值**

- `0`：成功
- 其他：失败

#### PutFrom

```c++
virtual int PutFrom(const std::string &key, void *buffer, size_t size, const int32_t direct = 3,
                        const ReplicateConfig &replicateConfig = {}) = 0;
```

**功能**

将 buffer 中的数据写入缓存并关联到 key。

**参数**

- `key`：数据键（长度 < 256字节）。
- `buffer`：目标内存地址。
- `size`：缓冲区容量。
- `direct`：数据流向。
- `replicateConfig`：副本策略配置。

**返回值**

- `0`：成功
- 其他：失败

#### Remove

```c++
virtual int Remove(const std::string &key) = 0;
```

**功能**

删除指定键的数据对象。

**参数**

`key`：数据键（长度 < 256字节）。

**返回值**

- `0`：成功
- 其他：失败

#### IsExist

```c++
virtual int IsExist(const std::string &key) = 0;
```

**功能**

检查键是否存在。

**参数**

`key`：数据键（长度 < 256字节）。

**返回值**

- `1`：存在
- `0`：不存在
- 其他：失败

#### GetKeyInfo

```c++
virtual KeyInfo GetKeyInfo(const std::string &key, uint32_t flag = 0) = 0;
```

**功能**

获取键的元信息。

**参数**

- `key`：数据的键，长度小于256个字节。
- `flag`：查询标志，默认值为 `0`。

**返回值**

返回 KeyInfo，包含：

- `size_`：数据字节数。
- `blobNum_`：数据副本数。
- `loc_`：数据副本所在位置列表。
- `type_`：数据副本所在介质类型列表。
- `gva_`：数据副本对应的 GVA 列表。

> [!NOTE] 补充说明
>
> - `GetKeyInfo` 的实际签名带 `flag` 参数。默认值为 `0`。
> - `KeyInfo` 除 `size_`、`blobNum_`、`loc_`、`type_` 外，还包含 `gva_`，用于描述每个 blob 对应的 GVA 列表。

### 4. 批量操作接口

#### BatchGetInto

```c++
virtual std::vector<int> BatchGetInto(const std::vector<std::string> &keys, const std::vector<void *> &buffers,
                                          const std::vector<size_t> &sizes, const int32_t direct = 2) = 0;
```

**功能**

批量读取多个键到各自缓冲区。

**参数**

- `keys`：数据键列表（每个键长度 < 256字节）。
- `buffers`：目标内存地址列表，必须与keys一一对应。
- `sizes`：缓冲区容量列表，必须与buffers长度一致。
- `direct`：数据流向。

**返回值**

返回每个键对应的处理结果列表，每个元素 `0` 表示成功，负数表示失败。

#### BatchPutFrom

```c++
virtual std::vector<int> BatchPutFrom(const std::vector<std::string> &keys, const std::vector<void *> &buffers,
                                          const std::vector<size_t> &sizes, const int32_t direct = 3,
                                          const ReplicateConfig &replicateConfig = {}) = 0;
```

**功能**

批量写入多个键。

**参数**

- `keys`：数据键列表（每个键长度 < 256字节）。
- `buffers`：目标内存地址列表，必须与keys一一对应。
- `sizes`：缓冲区容量列表，必须与buffers长度一致。
- `direct`：数据流向。
- `replicateConfig`：副本策略配置。

**返回值**

返回每个键对应的处理结果列表，每个元素 `0` 表示成功，负数表示失败。

#### BatchRemove

```c++
virtual std::vector<int> BatchRemove(const std::vector<std::string> &keys) = 0;
```

**功能**

批量删除。

**参数**

`keys`：数据键列表（每个键长度 < 256字节）。

**返回值**

返回每个键对应的处理结果列表，每个元素 `0` 表示成功，负数表示失败。

#### BatchIsExist

```c++
virtual std::vector<int> BatchIsExist(const std::vector<std::string> &keys) = 0;
```

**功能**

批量存在性检查。

**参数**

`keys`：数据键列表（每个键长度 < 256字节）。

**返回值**

- `1`：存在
- `0`：不存在
- 其他：失败

#### BatchGetKeyInfo

```c++
virtual std::vector<KeyInfo> BatchGetKeyInfo(const std::vector<std::string> &keys, uint32_t flag = 0) = 0;
```

**功能**

批量查询元信息。

**参数**

- `keys`：数据键列表（每个键长度 < 256字节）。
- `flag`：查询标志，默认值为 `0`。

**返回值**

返回KeyInfo列表，每个KeyInfo包含：

- `size_`：数据字节数。
- `blobNum_`：数据副本数。
- `loc_`：数据副本所在位置列表。
- `type_`：数据副本所在介质类型列表。
- `gva_`：数据副本对应的 GVA 列表。

#### PrefetchKeys

```c++
virtual int PrefetchKeys(std::vector<std::string> keys, uint8_t srcMedia = MMC_MEDIA_SSD,
                         uint8_t dstMedia = MMC_MEDIA_DRAM, uint32_t flags = 0) = 0;
```

**功能**

将指定 keys 从源介质回温到目标介质。当前仅支持 SSD→DRAM（`MMC_MEDIA_SSD` → `MMC_MEDIA_DRAM`），其他 src/dst 组合暂不支持，会返回 `MMC_INVALID_PARAM`。

**参数**

- `keys`：数据键列表（每个键长度 < 256字节）。
- `srcMedia`：源介质类型（`MMC_MEDIA_HBM`=0/`MMC_MEDIA_DRAM`=1/`MMC_MEDIA_SSD`=2），默认 `MMC_MEDIA_SSD`，预留参数。
- `dstMedia`：目标介质类型，默认 `MMC_MEDIA_DRAM`，预留参数。
- `flags`：预留标志，必须为 `0`。

**返回值**

- `0`：成功
- 其他：失败

#### EvictKeys

```c++
virtual int EvictKeys(std::vector<std::string> keys, uint8_t srcMedia = MMC_MEDIA_DRAM,
                      uint8_t dstMedia = MMC_MEDIA_SSD, uint32_t flags = 0) = 0;
```

**功能**

将指定 keys 从源介质驱逐到目标介质。当前仅支持 DRAM→SSD（`MMC_MEDIA_DRAM` → `MMC_MEDIA_SSD`），其他 src/dst 组合暂不支持，会返回 `MMC_INVALID_PARAM`。当 SSD 不可用或 DRAM 使用率高于回温水线时，DRAM 上的 blob 会被直接删除。

**参数**

- `keys`：数据键列表（每个键长度 < 256字节）。
- `srcMedia`：源介质类型（`MMC_MEDIA_HBM`=0/`MMC_MEDIA_DRAM`=1/`MMC_MEDIA_SSD`=2），默认 `MMC_MEDIA_DRAM`，预留参数。
- `dstMedia`：目标介质类型，默认 `MMC_MEDIA_SSD`，预留参数。
- `flags`：预留标志，必须为 `0`。

**返回值**

- `0`：成功
- 其他：失败

#### BatchAddLease

```c++
virtual std::vector<int> BatchAddLease(const std::vector<std::string> &keys, uint64_t leaseTtlMs = 0) = 0;
```

**功能**

批量为多个 key 增加读租约，并记录后续 GVA 读取所需的读租约状态。

**参数**

- `keys`：要增加读租约的 key 列表（每个键长度 < 256字节），不能为空。
- `leaseTtlMs`：要增加的租约时间，单位为毫秒，默认为 `0`。为 `0` 时使用 meta 侧配置项 `ock.mmc.meta.lease_ttl_ms`。

**返回值**

`std::vector<int>`：每个元素表示对应 key 的处理结果，`0` 表示成功，其他值表示失败。返回列表长度与 `keys` 一致。

**使用说明**

- 该接口返回每个 key 的错误码，不返回 `KeyInfo`。
- 该接口为非事务接口；某个 key 失败不会回滚其他 key 已经成功增加的读租约。
- `leaseTtlMs` 为 `0` 时，meta 侧使用配置项 `ock.mmc.meta.lease_ttl_ms` 的值增加租约。
- 调用方应只对返回值为 `0` 的 key 继续执行后续基于 GVA 的读取流程。
- 典型用法是先通过 `BatchGetKeyInfo(keys)` 获取 GVA，再调用 `BatchAddLease(keys)`，为后续基于 GVA 的读取流程建立读租约状态。
- 同一进程对同一 key 重复调用时，会复用当前进程中已有的读租约并在 meta 侧续租；完成读取后调用一次 `BatchRemoveLease` 即可释放。

#### BatchRemoveLease

```c++
virtual int BatchRemoveLease(const std::vector<std::string> &keys) = 0;
```

**功能**

批量移除多个 key 的读租约，并清理当前进程中对应的 GVA 读取状态。

**参数**

`keys`：要移除读租约的 key 列表（每个键长度 < 256字节），不能为空。

**返回值**

- `int`：`0` 表示本地读租约检查通过并已触发移除租约请求发送流程。
- 其他值表示失败。

**使用说明**

调用方完成基于 GVA 的读取后，应调用该接口显式释放由 `BatchAddLease` 建立的读租约。

#### BatchMalloc

```c++
virtual std::vector<uintptr_t> BatchMalloc(const std::vector<std::string> &keys, const std::vector<size_t> &sizes,
                                               uint16_t media, uint64_t leaseTtlMs = 0) = 0;
```

**功能**

批量为多个 key 申请全局内存，并返回每个 key 对应的起始 GVA。

**参数**

- `keys`：要申请内存的 key 列表，长度必须与 `sizes` 一致。
- `sizes`：每个 key 对应的数据大小列表。
- `media`：申请的介质类型，如 `MEDIA_HBM` 或 `MEDIA_DRAM`。
- `leaseTtlMs`：要增加的租约时间，单位为毫秒，默认为 `0`。为 `0` 时使用 meta 侧配置项 `ock.mmc.meta.lease_ttl_ms`。

**返回值**

- `std::vector<uintptr_t>`：每个元素为一个 key 对应的起始 GVA。
- 申请失败或参数非法时，对应元素为 `0`。

#### BatchCopy

```c++
virtual int BatchCopy(std::vector<void *> &gvas, std::vector<void *> &buffers, std::vector<size_t> &sizes,
                          const int32_t direct = 3) = 0;
```

**功能**

批量在 GVA 地址与本地缓冲区之间执行数据拷贝。

> [!NOTE] 说明
> 写方向（如 `SMEMB_COPY_L2G`/`SMEMB_COPY_H2G`）只拷贝数据，**不会**将 GVA 对应的 blob 状态翻转为 READABLE；调用方必须再调用 `MmcacheStore::BatchWriteFinish` 显式通知写完成，blob 才会进入可读状态。

**参数**

- `gvas`：GVA 地址列表。
- `buffers`：本地缓冲区列表，必须与 `gvas` 一一对应。
- `sizes`：每次拷贝的大小列表，长度必须与 `gvas` 一致。
- `direct`：数据拷贝方向，取值参见 `smem_bm_copy_type`，常用值包括 `SMEMB_COPY_L2G`、`SMEMB_COPY_G2L`、`SMEMB_COPY_G2H`、`SMEMB_COPY_H2G`。

**返回值**

- `0`：成功
- 其他：失败

#### BatchWriteFinish

```c++
virtual std::vector<int> BatchWriteFinish(const std::vector<std::string> &keys,
                                         const std::vector<int32_t> &writeResults) = 0;
```

**功能**

显式通知 meta service 给定 key 的写入已完成。

> [!NOTE] 说明
> 调用方在 `BatchCopy` 写方向完成后必须调用此接口，meta service 才会将对应 blob 从 `ALLOCATED` 翻转为 `READABLE`，此后其他进程才能通过 `BatchGetKeyInfo` / `BatchCopy` 读方向读到数据。对于已处于 `READABLE` 的 key，再次调用为幂等并直接返回成功；对未分配过的 key 返回 `MMC_UNMATCHED_KEY`。

**参数**

- `keys`：已写入完成的键列表，每个键长度小于 256 个字节。
- `writeResults`：与 `keys` 等长的每键写入结果，`0` 表示成功（对应 `MMC_WRITE_OK`，blob 翻为 `READABLE`），非 `0` 表示失败（对应 `MMC_WRITE_FAIL`，meta service 会移除该 blob）。

**返回值**

`std::vector<int>`：与 `keys` 等长，每个元素为 meta service 对该键的实际更新结果；`0` 表示成功。

> [!NOTE] 典型 GVA 跨进程读取流程：
>
> - 写进程：`BatchMalloc -> BatchCopy`（写入数据）-> `BatchWriteFinish` 显式通知写完成，blob 翻为 READABLE。
> - 读进程：`BatchGetKeyInfo(keys, 0)` 获取 GVA -> `BatchAddLease(...)` 增加读租约并准备 GVA 读取状态 ->  `BatchCopy`（读取）-> `BatchRemoveLease(...)` 显式释放读租约。

### 5. 分层张量操作

#### PutFromLayers

```c++
virtual int PutFromLayers(const std::string &key, const std::vector<void *> &buffers,
                              const std::vector<size_t> &sizes, const int32_t direct = 3,
                              const ReplicateConfig &replicateConfig = {}) = 0;

```

**功能**

将多个内存块（layers）拼接后作为一个逻辑对象写入缓存，并关联到指定键。

**参数**

- `key`：数据键（长度 < 256字节）。
- `buffers`：多层内存地址列表。
- `sizes`：每层缓冲区容量大小列表，必须与buffers长度一致。
- `direct`：数据流向。
- `replicateConfig`：副本策略配置。

**返回值**

- `0`：成功
- 其他：失败

#### GetIntoLayers

```c++
virtual int GetIntoLayers(const std::string &key, const std::vector<void *> &buffers,
                              const std::vector<size_t> &sizes, const int32_t direct = 2) = 0;
```

**功能**

从缓存中读取指定键的逻辑对象，并按预定义大小分发到多个目标缓冲区。

**参数**

- `key`：数据键（长度 < 256字节）。
- `buffers`：多层内存地址列表。
- `sizes`：每层缓冲区容量大小列表，必须与buffers长度一致。
- `direct`：数据流向。

**返回值**

- `0`：成功
- 其他：失败

#### BatchPutFromLayers

```c++
virtual std::vector<int> BatchPutFromLayers(const std::vector<std::string> &keys,
                                                const std::vector<std::vector<void *>> &buffers,
                                                const std::vector<std::vector<size_t>> &sizes, const int32_t direct = 3,
                                                const ReplicateConfig &replicateConfig = {}) = 0;
```

**功能**

批量将多个逻辑对象（每个由多层内存块组成）写入缓存。

**参数**

- `keys`：数据键列表（每个键长度 < 256字节）。
- `buffers`：多层内存地址列表，必须与keys一一对应。
- `sizes`：每层缓冲区容量大小列表，必须与buffers长度一致。
- `direct`：数据流向。
- `replicateConfig`：副本策略配置。

**返回值**

返回每个键对应的处理结果列表，每个元素 `0` 表示成功，负数表示失败

#### BatchGetIntoLayers

```c++
virtual std::vector<int> BatchGetIntoLayers(const std::vector<std::string> &keys,
                                                const std::vector<std::vector<void *>> &buffers,
                                                const std::vector<std::vector<size_t>> &sizes,
                                                const int32_t direct = 2) = 0;
```

**功能**

批量从缓存中读取多个逻辑对象，并分别分发到各自的多层缓冲区。

**参数**

- `keys`：数据键列表（每个键长度 < 256字节）。
- `buffers`：多层内存地址列表，必须与keys一一对应。
- `sizes`：每层缓冲区容量大小列表，必须与buffers长度一致。
- `direct`：数据流向。

**返回值**

返回每个键对应的处理结果列表，每个元素 `0` 表示成功，负数表示失败。

### 6. 辅助接口

#### GetLocalServiceId

```c++
virtual int GetLocalServiceId(uint32_t &localServiceId) = 0;
```

**功能**

获取当前实例关联的本地服务 ID（用于调试或日志追踪）。

**参数**

`localServiceId`：输出参数。

**返回值**

- `0`：成功
- 其他：失败

## 数据结构

### ReplicateConfig

副本策略配置，包含以下字段：

- `replicaNum`：副本数，默认1，最大8。
- `preferredLocalServiceIDs`：优先分配的本地服务 ID 列表，列表大小必须小于或等于replicaNum。

### KeyInfo

键元信息结构体，包含以下字段：

- `size_`：数据字节数。
- `blobNum_`：数据副本数。
- `loc_`：数据副本所在位置列表。
- `type_`：数据副本所在介质类型列表。
- `gva_`：数据副本的全局虚拟地址列表。

## smem_bm_copy_type 枚举类型

| 类型 | 值 | 说明 |
| ------------------- | ---- | -------------------- |
| SMEMB_COPY_L2G | 0 | 从卡上内存复制到全局内存 |
| SMEMB_COPY_G2L | 1 | 从全局内存复制到卡上内存 |
| SMEMB_COPY_G2H | 2 | 从全局内存复制到主机内存 |
| SMEMB_COPY_H2G | 3 | 从主机内存复制到全局内存 |
| SMEMB_COPY_L2GH | 4 | 从卡上内存复制到全局主机内存 |
| SMEMB_COPY_GH2L | 5 | 从全局主机内存复制到卡上内存 |
| SMEMB_COPY_GH2H | 6 | 从全局主机内存复制到主机内存 |
| SMEMB_COPY_H2GH | 7 | 从主机内存复制到全局主机内存 |
| SMEMB_COPY_G2G | 8 | 从全局内存复制到全局内存 |
| SMEMB_COPY_AUTO | 9 | 自动选择数据拷贝方向 |

## 错误码

| 值     | 说明        |
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
| -3105 | 元数据备份失败   |

## 注意事项

- 所有键的长度必须小于256个字节。
- 支持同步和异步两种操作模式。
- 批量操作可以提高处理效率。


### 可选具体实现扩展：`MmcacheStore::BatchAddLeaseWithInfo`

```cpp
std::pair<std::vector<int>, std::vector<KeyInfo>> BatchAddLeaseWithInfo(
    const std::vector<std::string> &keys, uint64_t leaseTtlMs = 0);
```

该 nonvirtual 方法使用同一次 Add 响应返回逐 key 状态及 KeyInfo；原 ObjectStore 虚表、
原 Add/Remove 接口、客户端对象数据布局和 Meta wire 协议保持不变。
需使用同一构建产出的 libmmc 和 Python binding，不可把新 binding 与旧 libmmc 混装。
错误后的租约所有权规则同 Python 扩展说明；该扩展不提供裸地址跨 step 缓存许可。
