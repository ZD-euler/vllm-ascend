"""Opt-in MMC deployment flag using the unchanged qualified MF1.3 allocator."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CONFIG='src/memcache/csrc/config/'


def once(text, old, new):
    if text.count(old)!=1:
        raise ValueError('qualified config patch site changed')
    return text.replace(old,new,1)


def patch(base):
    names=[CONFIG+'mmc_config_const.h',CONFIG+'mmc_configuration.h',
        'docs/zh/memcache_config.md','test/ut/testcase/memcache/csrc/config/test_mmc_configuration.cpp']
    before={name:(base/name).read_text() for name in names}
    after=dict(before)
    old='constexpr auto OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT ='
    after[names[0]]=once(after[names[0]],old,'''// Opt-in ordinary 2MiB DRAM through MF1.3 HYBM_FLAG_UNRESTRICTED_MEM.
constexpr auto OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED =
    std::make_pair("ock.mmc.local_service.dram.unrestricted.enabled", false);

'''+old)
    old='constexpr uint32_t CONF_MUST = 1;'
    after[names[1]]=once(after[names[1]],old,old+'''
// Existing MF1.3 HyBM flag; public request/config struct layouts are unchanged.
constexpr uint32_t MMC_BM_FLAG_DRAM_UNRESTRICTED = (1U << 11);''')
    old='        AddBoolConf(OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT,'
    after[names[1]]=once(after[names[1]],old,'''        AddBoolConf(OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED,
                    VStrEnum::Create(OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED.first, BOOL_ENUM_STR), 0);
'''+old)
    old='        if (GetBool(ConfConstant::OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT)) {'
    after[names[1]]=once(after[names[1]],old,'''        if (GetBool(ConfConstant::OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED)) {
            config.flags |= MMC_BM_FLAG_DRAM_UNRESTRICTED;
        }
'''+old)
    old='        std::string protocol(config.dataOpType);'
    after[names[1]]=once(after[names[1]],old,old+'''
        if ((config.flags & MMC_BM_FLAG_DRAM_UNRESTRICTED) != 0U && protocol != "device_sdma") {
            MMC_LOG_ERROR("dram.unrestricted requires device_sdma, protocol:" << protocol);
            return MMC_INVALID_PARAM;
        }''')
    after[names[2]]+='''

### Opt-in ordinary DDR allocation for `device_sdma`

`ock.mmc.local_service.dram.unrestricted.enabled` defaults to `false`.
When explicitly `true`, MMC passes MF1.3's existing `HYBM_FLAG_UNRESTRICTED_MEM`
to use ordinary 2MiB DDR rather than the default P2P-DDR allocation path.
Other protocols are rejected when this option is enabled. Existing request,
config-struct and operator interfaces do not change; the default path is unchanged.
The ordinary-DDR branch does not honor NUMA flag bits, so this option does not
promise a particular NUMA placement. Qualify real multi-device SDMA/MTE and
model correctness before accepting this deployment configuration.
'''
    after[names[3]]+='''

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
'''
    return before,after


def main(base):
    before,after=patch(base)
    files={};unified=[]
    for name,text in after.items():
        path=HERE/'overlay'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)
        files[name]=dict(base_sha256=hashlib.sha256(before[name].encode()).hexdigest(),
                        candidate_sha256=hashlib.sha256(text.encode()).hexdigest())
        unified.extend(difflib.unified_diff(before[name].splitlines(True),text.splitlines(True),
            fromfile='a/'+name,tofile='b/'+name))
    (HERE/'manifest.json').write_text(json.dumps(dict(schema='MMC_optin_unrestricted_DRAM_v17',files=files,
        default_enabled=False,existing_struct_layouts_unchanged=True,MF_modified=False,NPU_qualified=False,
        caveat='MF ordinary-DDR branch ignores NUMA flag bits; actual placement is not pinned to node0.'),indent=2)+'\n')
    (HERE/'unrestricted-v17.patch').write_text(''.join(unified))
    print(json.dumps(dict(status='optin_overlay_created',config_files=2,doc_files=1,UT_files=1,
        MF_modified=False,NPU_qualified=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args();main(args.baseline.resolve(strict=True))
