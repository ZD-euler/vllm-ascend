"""Compile actual patched config methods and unchanged MF allocator decisions.

Minimal collaborators are fake; this proves flag/default/decision contracts,
not a real SDK parser build, HAL allocation, SDMA, MTE or SFA correctness.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent


def function(text, signature):
    start=text.index(signature);begin=text.index('{',start);end=begin+1;depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]


def run(base, mf):
    config=(HERE/'overlay/src/memcache/csrc/config/mmc_configuration.h').read_text()
    host=(mf/'src/hybm/csrc/mm/hybm_vmm_based_segment.cpp').read_text()
    smem=(mf/'src/smem/csrc/smem_bm/smem_bm.cpp').read_text()
    public=(mf/'src/hybm/include/hybm_def.h').read_text()
    assert 'options.flags = option->flags;' in smem
    assert 'hybmOpts.flags &= (~HYBM_FLAG_CREATE_WITH_SHM);' in smem
    assert '#define HYBM_FLAG_UNRESTRICTED_MEM   (1U << 11)' in public
    get=function(config,'    void GetLocalServiceConfig(')
    validate=function(config,'    static Result ValidateLocalServiceConfig(')
    adapter=function(host,'Result HybmVmmBasedSegment::HalMemCreateAdapterFromHost(')
    malloc=function(host,'Result HybmVmmBasedSegment::MallocFromHost(')
    registration=config[config.index('        AddBoolConf(OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED,'):]
    registration=registration[:registration.index(');')+2]
    config_flag=next(line for line in config.splitlines() if line.startswith('constexpr uint32_t MMC_BM_FLAG_DRAM_UNRESTRICTED'))
    code=r'''
#include <algorithm>
#include <cassert>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include "mmc_config_const.h"
using namespace ock::mmc;using namespace ock::mmc::ConfConstant;
#define MMC_LOG_DEBUG(...) ((void)0)
#define MMC_LOG_ERROR(...) ((void)0)
#define MMC_LOG_INFO(...) ((void)0)
#define BM_LOG_INFO(...) ((void)0)
#define BM_LOG_ERROR(...) ((void)0)
#define BM_LOG_WARN(...) ((void)0)
#define MMC_VALIDATE_RETURN(c,m,r) do {if(!(c))return r;}while(0)
using Result=int;constexpr int MMC_OK=0,MMC_INVALID_PARAM=-2,BM_ERROR=-1,BM_INVALID_PARAM=-2,BM_OK=0;
constexpr int HAL_OUT_OF_MEMORY_ERROR=6;
CONFIG_FLAG_DECL
constexpr uint32_t SMEM_BM_FLAG_DRAM_BEST_EFFORT=(1U<<10);
constexpr uint32_t HYBM_FLAG_UNRESTRICTED_MEM=(1U<<11),UINT32_WIDTH=32,HYBM_BIND_NUMA_FLAG_INDEX=0,HYBM_BIND_NUMA_FLAG_LEN=7;
constexpr uint64_t DRAM_SIZE_ALIGNMENT=2ULL<<20;
constexpr int DISCOVERY_URL_SIZE=256,MMC_TAG_SIZE=256,MMC_TAG_OP_INFO_SIZE=256,PROTOCOL_SIZE=256;
struct mmc_local_service_config_t {
 char discoveryURL[256]{},bmIpPort[256]{},bmIpPortDomain[256]{},bmHcomUrl[256]{},backendId[256]{},
 dataOpType[256]{},tag[256]{},tagOpInfo[256]{};
 uint32_t worldSize{},createId{},flags{},dynamicConfigInterval{};
 uint64_t localDRAMSize{},localMaxDRAMSize{},localHBMSize{},localMaxHBMSize{};
 bool storageEnabled{},dynamicConfigEnable{};int logLevel{},accTlsConfig{},hcomTlsConfig{},configStoreTlsConfig{};
};
static void SafeCopy(const std::string &a,char*b,size_t n){std::strncpy(b,a.c_str(),n-1);b[n-1]=0;}
static void StringToUpper(std::string &a){std::transform(a.begin(),a.end(),a.begin(),::toupper);}
struct MmcOutLogger {static MmcOutLogger &Instance(){static MmcOutLogger a;return a;}int GetLogLevel(const std::string&){return 0;}};
struct VStrEnum {static int Create(const std::string&,const std::string&p){assert(p=="false||true");return 0;}};
struct Registration {bool found=false;void AddBoolConf(const std::pair<const char*,bool>&p,int,int){assert(!p.second);found=true;}
 void check(){const std::string BOOL_ENUM_STR="false||true";REGISTRATION}}
;
struct FixtureConfig {
 bool enabled=false,best=false;std::string protocol="device_sdma";
 bool GetBool(const std::string &k){return k==OCK_MMC_LOCAL_SERVICE_DRAM_UNRESTRICTED.first?enabled:k==OCK_MMC_LOCAL_SERVICE_DRAM_BEST_EFFORT.first?best:false;}
 std::string GetString(const std::string &k){return k==OKC_MMC_LOCAL_SERVICE_PROTOCOL.first?protocol:k==OCK_MMC_LOG_LEVEL.first?"error":"";}
 int GetInt(const std::string&){return 4;}
 template<class K,class V>bool GetBool(const std::pair<K,V>&p){return GetBool(std::string(p.first));}
 template<class K,class V>std::string GetString(const std::pair<K,V>&p){return GetString(std::string(p.first));}
 template<class K,class V>int GetInt(const std::pair<K,V>&p){return GetInt(std::string(p.first));}
 uint64_t GetUInt64(const std::string&k,uint64_t d){return k.find("dram.size")!=std::string::npos?(10ULL<<30):d;}
 void GetAccTlsConfig(int&){}void GetHcomTlsConfig(int&){}void GetConfigStoreTlsConfig(int&){}
 static int ValidateTLSConfig(int){return 0;}
 GET_CONFIG
 VALIDATE_CONFIG
};
enum class AscendSocType {ASCEND_910C,ASCEND_950};
enum class NumaBindPolicy {OFF,AUTO,MANUAL};
struct NumaInfo {bool valid=true;NumaBindPolicy policy=NumaBindPolicy::OFF;uint32_t numaIndex=0;};
struct HybmNumaUtil {static NumaInfo GetNumaBindPolicyInfo(uint32_t f){return f&128?NumaInfo{true,NumaBindPolicy::MANUAL,f&127}:NumaInfo{};}};
enum {MEM_HOST_SIDE=0,MEM_HOST_NUMA_SIDE=2,MEM_HUGE_PAGE_TYPE=1,MEM_GIANT_PAGE_TYPE=2,MEM_DDR_TYPE=1,MEM_P2P_DDR_TYPE=3,MEM_ALLOC_GRANULARITY_RECOMMENDED=1};
struct drv_mem_handle_t{};struct drv_mem_prop {uint32_t side,devid,slot,pg_type,mem_type,reserved;};
std::vector<drv_mem_prop> called;bool failGiant=false;
struct DlAclApi {static AscendSocType GetAscendSocType(){return AscendSocType::ASCEND_910C;}};
struct DlHalApi {
 static int HalMemGetAllocationGranularity(const drv_mem_prop*,int,size_t*g){*g=1ULL<<30;return 0;}
 static int HalMemCreate(drv_mem_handle_t**,size_t,const drv_mem_prop*p,int){called.push_back(*p);return failGiant&&p->pg_type==MEM_GIANT_PAGE_TYPE?6:0;}
};
struct HybmVmmBasedSegment {struct {uint32_t flags=0;}options_;
 Result HalMemCreateAdapterFromHost(size_t,drv_mem_handle_t**,drv_mem_prop);
 Result MallocFromHost(size_t,uint32_t,drv_mem_handle_t**) noexcept;
};
HOST_ADAPTER
HOST_MALLOC
int main(){
 Registration registration;registration.check();assert(registration.found);
 FixtureConfig cfg;mmc_local_service_config_t out{};cfg.GetLocalServiceConfig(out);assert(out.flags==0);
 assert(FixtureConfig::ValidateLocalServiceConfig(out)==0); // default contract unchanged
 cfg.best=true;out={};cfg.GetLocalServiceConfig(out);assert(out.flags==1024);
 cfg.best=false;cfg.enabled=true;out={};cfg.GetLocalServiceConfig(out);assert(out.flags==2048);
 assert(FixtureConfig::ValidateLocalServiceConfig(out)==0&&out.localDRAMSize==(10ULL<<30));
 cfg.protocol="host_rdma";out={};cfg.GetLocalServiceConfig(out);assert(FixtureConfig::ValidateLocalServiceConfig(out)==-2);
 cfg.enabled=false;out={};cfg.GetLocalServiceConfig(out);assert(out.flags==0&&FixtureConfig::ValidateLocalServiceConfig(out)==0);
 HybmVmmBasedSegment segment;drv_mem_handle_t *h=nullptr;called.clear();
 assert(segment.MallocFromHost(10ULL<<30,0,&h)==0);assert(called.size()==1&&called[0].pg_type==MEM_GIANT_PAGE_TYPE&&called[0].mem_type==MEM_P2P_DDR_TYPE);
 segment.options_.flags=2048;called.clear();
 assert(segment.MallocFromHost(10ULL<<30,0,&h)==0);assert(called.size()==1&&called[0].pg_type==MEM_HUGE_PAGE_TYPE&&called[0].mem_type==MEM_DDR_TYPE&&called[0].side==MEM_HOST_SIDE);
 segment.options_.flags=2048|128|1;called.clear();
 assert(segment.MallocFromHost(10ULL<<30,0,&h)==0);assert(called[0].devid==0&&called[0].side==MEM_HOST_SIDE); // NUMA bits intentionally ignored by unchanged MF
 segment.options_.flags=128|1;called.clear();
 assert(segment.MallocFromHost(10ULL<<30,0,&h)==0);assert(called[0].devid==1&&called[0].mem_type==MEM_P2P_DDR_TYPE);
 segment.options_.flags=0;called.clear();failGiant=true;
 assert(segment.MallocFromHost(10ULL<<30,0,&h)==0&&called.size()==2&&called.back().pg_type==MEM_HUGE_PAGE_TYPE&&called.back().mem_type==MEM_P2P_DDR_TYPE);
 return 0;
}
'''.replace('CONFIG_FLAG_DECL',config_flag).replace('REGISTRATION',registration).replace('GET_CONFIG',get).replace('VALIDATE_CONFIG',validate).replace('HOST_ADAPTER',adapter).replace('HOST_MALLOC',malloc)
    with tempfile.TemporaryDirectory() as temp:
        directory=Path(temp);source=directory/'actual_functions.cpp';source.write_text(code)
        include=HERE/'overlay/src/memcache/csrc/config'
        built=subprocess.run(['c++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-Wno-unused-variable',
            '-I'+str(include),str(source),'-o',str(directory/'test')],capture_output=True,text=True)
        if built.returncode:
            raise RuntimeError(built.stderr[:3500])
        subprocess.run([str(directory/'test')],check=True)
    print(json.dumps(dict(status='CPU_actual_function_contracts_passed',config_cases=6,MF_allocator_cases=5,
        default_unchanged=True,NUMA_ignored_when_unrestricted=True,real_SDK_build=False,NPU_initialized=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--mf',type=Path,required=True)
    args=parser.parse_args();run(args.baseline.resolve(strict=True),args.mf.resolve(strict=True))
