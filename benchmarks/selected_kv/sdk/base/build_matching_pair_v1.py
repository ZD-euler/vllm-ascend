"""CPU-build the frozen official C-ABI pair into an independent owned overlay.

No pip install, SDK import, NPU initialization, global configuration or system
library replacement occurs. All source hashes are checked before configure.
Original official CMake targets perform the build and owned-prefix install.
Successful compilation is not an SDK lifecycle or payload-correctness gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for value in iter(lambda:stream.read(1024*1024),b''):digest.update(value)
    return digest.hexdigest()


def verify(sources,manifest):
    if manifest['major_minor']!='1.3':raise ValueError('Expected the reviewed C ABI 1.3 source pair')
    actual=set()
    for role in ('mmc','mf'):
        for path in (sources/role).rglob('*'):
            if path.is_symlink():raise ValueError('Source archive must contain regular source files only')
            if path.is_file():actual.add(str(path.relative_to(sources)))
    if actual!=set(manifest['files']):raise ValueError('Source archive has missing or unmanifested files')
    for name,expected in manifest['files'].items():
        path=sources/name
        if not path.is_file() or not path.resolve().is_relative_to(sources) or sha(path)!=expected:
            raise ValueError('Frozen public source hash mismatch')
    for role in ('mmc','mf'):
        if (sources/role/'VERSION').read_text().strip()!='1.3.0':raise ValueError('Source ABI version differs')


def one(root,pattern):
    found=sorted(root.rglob(pattern))
    if len(found)!=1:raise ValueError('Expected one compiled artifact: '+pattern)
    return found[0]


def copy_file(source,destination):
    if not source.is_file() or destination.exists():raise ValueError('Artifact missing or duplicate')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,destination)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--build-aicpu',action='store_true')
    args=parser.parse_args()
    sources=args.sources.resolve(strict=True)
    output=args.out.absolute()
    python=args.python.absolute()  # Preserve a venv symlink's invocation path.
    if not python.is_file() or not 1<=args.jobs<=32:raise ValueError('Explicit Python and bounded build parallelism required')
    if output.exists() or not output.parent.is_dir():raise ValueError('Use a fresh owned output directory')
    manifest=json.loads(args.manifest.read_text())
    verify(sources,manifest)
    output.mkdir()
    (output/'logs').mkdir()
    state=dict(status='verified_public_sources',SDK_imported=False,NPU_initialized=False,
        installed_SDK_modified=False,SDK_runtime_qualified=False,source_pair={key:manifest[key] for key in ('MMC_commit','MF_commit','major_minor')},
        source_file_count=len(manifest['files']),stages=[],aicpu_requested=args.build_aicpu)

    def publish():
        temporary=output/'public-build.next.json'
        temporary.write_text(json.dumps(state,indent=2)+'\n')
        temporary.replace(output/'public-build.json')
    publish()
    env=dict(os.environ)
    env['PATH']=str(python.parent)+os.pathsep+env.get('PATH','')
    env['MF_BUILD_JOBS']=str(args.jobs)
    env['MMC_BUILD_JOBS']=str(args.jobs)

    def run(stage,command,*,cwd=None,timeout=1200):
        state['status']=stage+'_running';publish()
        begin=time.monotonic()
        with (output/'logs'/(stage+'.log')).open('w') as log:
            result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
        state['stages'].append(dict(stage=stage,returncode=result.returncode,wall_seconds=time.monotonic()-begin))
        publish()
        if result.returncode:raise RuntimeError('Public SDK build stage failed: '+stage)

    try:
        # Check only pure Python build dependencies. Do not import SDK/torch/ACL.
        run('python_build_dependencies',[str(python),'-c','import pybind11,sysconfig; assert sysconfig.get_path("include")'],timeout=30)
        generator=['-G','Ninja'] if shutil.which('ninja',path=env['PATH']) else []
        prefix=output/'prefix';prefix.mkdir()
        for role in ('mf','mmc'):
            (prefix/role).mkdir()
            build=output/('build-'+role)
            # MF's own Python CMake files explicitly force ABI0. Its native
            # libraries must therefore use ABI0 too. MMC's nested spdlog CMake
            # uses default ABI1, so MMC uses ABI1. The reviewed new pure C ABI
            # is precisely the supported boundary between these two builds.
            abi='OFF' if role=='mf' else 'ON'
            flags=['-DCMAKE_BUILD_TYPE=RELEASE','-DBUILD_PYTHON=ON','-DBUILD_OPEN_ABI='+abi,
                '-DBUILD_GIT_COMMIT=OFF','-DENABLE_PTRACER=OFF','-DBUILD_CPP_EXAMPLE=OFF',
                '-DFETCHCONTENT_FULLY_DISCONNECTED=ON',
                '-DCMAKE_INSTALL_PREFIX='+str(prefix/role),'-DPYTHON_EXECUTABLE='+str(python)]
            if role=='mf':
                flags+=['-DXPU_TYPE=NPU','-DBUILD_UT=OFF','-DBUILD_TEST=OFF',
                        '-DBUILD_HCOM=OFF','-DBUILD_ETCD_BACKEND=OFF']
            else:
                flags+=['-DBUILD_TESTS=OFF','-DMMC_ENABLE_KV_EVENTS=OFF','-DBUILD_UBSIO=OFF']
            run(role+'_configure',['cmake','-S',str(sources/role),'-B',str(build),*generator,*flags],timeout=300)
            run(role+'_build_install',['cmake','--build',str(build),'--target','install','--parallel',str(args.jobs)])
        mf,mmc=sources/'mf',sources/'mmc'
        if args.build_aicpu:
            run('mf_aicpu_compile',['bash',str(mf/'script/kernel/hybm/build.sh')],cwd=mf)
            names=('cann-hybm-compat.tar.gz','libcann_hybm_kernel.json','cann_hybm_kernel_version','install.sh')
            if not all((mf/'output/hybm/aicpu_kernel'/name).is_file() for name in names):
                raise RuntimeError('Official kernel script skipped required AICPU artifacts')
        site=output/'site';site.mkdir()
        shutil.copytree(mf/'src/smem/python/memfabric_hybrid/memfabric_hybrid',site/'memfabric_hybrid',
                        ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(mmc/'src/memcache/python/memcache_hybrid',site/'memcache_hybrid',
                        ignore=shutil.ignore_patterns('__pycache__'))
        for name,component in [('libmf_hybm_core.so','hybm'),('libmf_smem.so','smem'),('libmf_acc_offload.so','acc_offload')]:
            copy_file(prefix/'mf'/component/'lib64'/name,site/'memfabric_hybrid/lib'/name)
        copy_file(prefix/'mmc/memcache/lib64/libmf_memcache.so',site/'memcache_hybrid/lib/libmf_memcache.so')
        for pattern in ('_pymf_hybrid*.so','_pymf_transfer*.so','_pymf_acc_offload*.so'):
            source=one(output/'build-mf',pattern);copy_file(source,site/'memfabric_hybrid'/source.name)
        source=one(output/'build-mmc','_pymmc*.so');copy_file(source,site/'memcache_hybrid'/source.name)
        shutil.copytree(mmc/'config',site/'memcache_hybrid/config',dirs_exist_ok=True)
        for component,relative in [('smem','smem'),('hybm','hybm')]:
            shutil.copytree(prefix/'mf'/component/'include',site/'memfabric_hybrid/include'/relative,dirs_exist_ok=True)
        if args.build_aicpu:
            shutil.copytree(mf/'output/hybm/aicpu_kernel',site/'memfabric_hybrid/_kernel/hybm')
        for package in ('memfabric_hybrid','memcache_hybrid'):
            dist=site/(package+'-1.3.0.dist-info');dist.mkdir()
            (dist/'METADATA').write_text('Metadata-Version: 2.1\nName: '+package.replace('_','-')+'\nVersion: 1.3.0\n')
            (dist/'INSTALLER').write_text('owned-reviewed-source-build; no pip/system install\n')
        required=set()
        for source in (mmc/'src/memcache/csrc/under_api/mf_smem').glob('*.cpp'):
            required.update(re.findall(r'"(smem_[A-Za-z0-9_]+)"',source.read_text()))
        if len(required)!=54:raise RuntimeError('The frozen MMC C ABI symbol manifest changed')
        smem=site/'memfabric_hybrid/lib/libmf_smem.so'
        symbols=subprocess.check_output(['nm','-D','--defined-only',str(smem)],text=True)
        exports={line.split()[-1] for line in symbols.splitlines() if line.split()}
        missing=sorted(required-exports)
        if missing:raise RuntimeError('Built MF is missing mandatory MMC C ABI symbols: '+','.join(missing))
        hybm=subprocess.check_output(['nm','-D','--defined-only',str(site/'memfabric_hybrid/lib/libmf_hybm_core.so')],text=True)
        if not re.search(r'\bhybm_gva_to_va\s*$',hybm,re.M):raise RuntimeError('Built HyBM lacks source mapping API')
        dynamic=subprocess.check_output(['readelf','-d',str(site/'memcache_hybrid/lib/libmf_memcache.so')],text=True)
        needed=re.findall(r'\(NEEDED\).*?\[([^\]]+)\]',dynamic)
        if any(name.startswith(('libmf_smem','libacc_tcp_net')) for name in needed):
            raise RuntimeError('MMC build still directly links the old C++ MF boundary')
        state.update(status='CPU_build_and_static_ABI_checks_passed',mandatory_C_ABI_symbol_count=len(required),
            MF_CXX11_ABI=0,MMC_CXX11_ABI=1,
            missing_C_ABI_symbols=[],hybm_gva_to_va_exported=True,MMC_direct_MF_DT_NEEDED=False,
            aicpu_artifacts_built=args.build_aicpu,
            overlay_files={str(p.relative_to(site)):sha(p) for p in site.rglob('*') if p.is_file()},
            environment_for_parent_only=dict(PYTHONPATH=str(site),
                LD_LIBRARY_PATH=str(site/'memfabric_hybrid/lib')+':'+str(site/'memcache_hybrid/lib')))
        publish()
    except Exception as error:
        state.update(status='failed',exception_type=type(error).__name__)
        publish()
        raise


if __name__=='__main__':main()
