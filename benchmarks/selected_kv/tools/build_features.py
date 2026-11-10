"""CPU compiler work only: build the exact candidate selected-KV libraries separately.

This builds the existing CMake targets, not a replacement wheel or baseline.
No framework/NPU import and no driver initialization is performed here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import signal
import time
from build_receipts import FEATURES, check_inputs, emit, install_build_signal_handlers

BUNDLE = Path(__file__).resolve().parents[1]

# Portable paths are supplied explicitly; extracted source only.
VERSION_LABEL = 'M3v15'
DEADLINE=time.time()+2400


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=BUNDLE.parents[1])
    parser.add_argument('--cann', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replace-feature-targets', action='store_true')
    args = parser.parse_args()
    install_build_signal_handlers()
    if args.output.resolve().is_relative_to(BUNDLE):
        raise ValueError('build output must stay outside the code bundle')
    os.environ['ASCEND_HOME_PATH'] = str(args.cann.resolve(strict=True))
    if DEADLINE-time.time()<300:raise TimeoutError('not enough authorized time for a new build')
    root = args.output.resolve()
    root.mkdir(exist_ok=False)
    source = args.source.resolve(strict=True)
    inputs = check_inputs(BUNDLE, 'selected_features', 'm3v9', source)
    for name in FEATURES:
        target = source / 'vllm_ascend/lib' / name
        if target.exists():
            if not args.replace_feature_targets:
                raise ValueError('existing feature output; use explicit --replace-feature-targets in owned source')
            target.unlink()
    torch = Path(importlib.util.find_spec('torch').origin).parent
    torch_npu = Path(importlib.util.find_spec('torch_npu').origin).parent
    cann = Path(os.environ['ASCEND_HOME_PATH'])
    cmake_entry = next((cann/p/'ascendc.cmake' for p in (
        'tools/tikcpp/ascendc_kernel_cmake', 'compiler/tikcpp/ascendc_kernel_cmake',
        'ascendc_devkit/tikcpp/samples/cmake') if (cann/p/'ascendc.cmake').is_file()), None)
    if cmake_entry is None:
        raise FileNotFoundError('CANN AscendC CMake entry is absent')
    wrapper = root/'wrapper'
    wrapper.mkdir()
    (wrapper/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.16)
project(exact_selected_kv_candidate LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(RUN_MODE npu)
set(SOC_VERSION ascend910_9382 CACHE STRING "Verified A3 compiler target")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
set(ASCEND_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)
include_directories(BEFORE "${ASCEND_HOME_PATH}/include" ${Python3_INCLUDE_DIRS})
include("'''+str(cmake_entry)+'''")
add_custom_target(vllm_ascend_C)
include("'''+str(source/'csrc/sparse_prefill_native/CMakeLists.txt')+'''")
''')
    report = dict(status='building', commit='db85ccd37d8fe90b9835903355f3bc92c1220e85', NPU_initialized=False,
                  product_python_changed=False, SOC_VERSION='ascend910_9382',
                  full_wheel_built=False, target_scope='existing selected-KV targets only')
    public = root/'public-exact-build.json'
    public.write_text(json.dumps(report, indent=2)+'\n')
    start = time.monotonic()
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', MAX_JOBS='2')
    commands = [
        ['cmake', '-S', str(wrapper), '-B', str(root/'build'),
         '-DCMAKE_PREFIX_PATH='+str(torch/'share/cmake'),
         '-DTORCH_NPU_PATH='+str(torch_npu),
         '-DVLLM_ASCEND_PACKAGE_PATH='+str(source/'vllm_ascend')],
        ['cmake', '--build', str(root/'build'), '--target', 'vllm_ascend_C', '-j2'],
        ['cmake', '--install', str(root/'build')],
    ]
    codes = []
    try:
        with (root/'private-build.log').open('x') as log:
            for cmd in commands:
                remaining=DEADLINE-time.time()-30
                if remaining<30:raise TimeoutError('finite build deadline reached')
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:code=child.wait(timeout=min(2400,remaining))
                except BaseException:
                    if child.poll() is None and os.getpgid(child.pid)==child.pid:
                        os.killpg(child.pid,signal.SIGTERM)
                        try:child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                    raise
                codes.append(code)
                if code:
                    raise RuntimeError('exact candidate CMake phase failed')
        outputs = list((source/'vllm_ascend/lib').glob('*'))
        report.update(status='compiled_not_NPU_validated',
                      artifacts={p.name:digest(p) for p in outputs if p.is_file()})
    except BaseException as error:
        report.update(status='failed', error_class=type(error).__name__)
    report.update(returncodes=codes, wall_seconds=time.monotonic()-start)
    public.write_text(json.dumps(report, indent=2)+'\n')
    if report['status'] == 'compiled_not_NPU_validated':
        emit(BUNDLE, 'selected_features', 'm3v9', source, source, inputs,
             [{'argv': command, 'exit_code': code} for command, code in zip(commands, codes)],
             root / 'receipt.json')
    print(json.dumps(report))
    return 0 if report['status']=='compiled_not_NPU_validated' else 1


if __name__ == '__main__':
    raise SystemExit(main())
