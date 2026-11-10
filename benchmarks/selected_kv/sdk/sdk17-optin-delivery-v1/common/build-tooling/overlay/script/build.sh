#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# MemCache_Hybrid is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

export BUILD_MODE=${1:-RELEASE}
export BUILD_TESTS=${2:-OFF}
export BUILD_OPEN_ABI=${3:-OFF}
export BUILD_PYTHON=${4:-ON}
export ENABLE_PTRACER=${5:-ON}
export INCREMENTAL=${6:-OFF}
export BUILD_UBSIO=${7:-ON}
BUILD_KV_EVENTS=${8:-}
BUILD_GIT_COMMIT=${9:-}
BUILD_OFFLINE=${10:-}
MMC_OPTION_ARGS=()
for option in BUILD_KV_EVENTS BUILD_GIT_COMMIT BUILD_OFFLINE; do
    value=${!option}
    if [[ -n "$value" && "$value" != "ON" && "$value" != "OFF" ]]; then
        echo "$option requires ON or OFF" >&2; exit 2
    fi
done
if [[ -n "$BUILD_KV_EVENTS" ]]; then MMC_OPTION_ARGS+=("-DMMC_ENABLE_KV_EVENTS=$BUILD_KV_EVENTS"); fi
if [[ -n "$BUILD_GIT_COMMIT" ]]; then MMC_OPTION_ARGS+=("-DBUILD_GIT_COMMIT=$BUILD_GIT_COMMIT"); fi
if [[ -n "$BUILD_OFFLINE" ]]; then MMC_OPTION_ARGS+=("-DFETCHCONTENT_FULLY_DISCONNECTED=$BUILD_OFFLINE"); fi

readonly SCRIPT_FULL_PATH=$(dirname $(readlink -f "$0"))
readonly PROJECT_FULL_PATH=$(dirname "$SCRIPT_FULL_PATH")
readonly MMC_BUILD_JOBS="${MMC_BUILD_JOBS:-32}"

if ! command -v dos2unix &> /dev/null; then
  dos2unix() { sed -i 's/\r$//' "$@"; }
fi

if [ "${BUILD_TESTS}" == "ON" ]; then
  readonly MOCKCPP_PATH="$PROJECT_FULL_PATH/test/3rdparty/mockcpp"
  readonly TEST_3RD_PATCH_PATH="$PROJECT_FULL_PATH/test/3rdparty/patch"
  dos2unix "$MOCKCPP_PATH/include/mockcpp/JmpCode.h"
  dos2unix "$MOCKCPP_PATH/include/mockcpp/mockcpp.h"
  dos2unix "$MOCKCPP_PATH/src/JmpCode.cpp"
  dos2unix "$MOCKCPP_PATH/src/JmpCodeArch.h"
  dos2unix "$MOCKCPP_PATH/src/JmpCodeX64.h"
  dos2unix "$MOCKCPP_PATH/src/JmpCodeX86.h"
  dos2unix "$MOCKCPP_PATH/src/JmpOnlyApiHook.cpp"
  dos2unix "$MOCKCPP_PATH/src/UnixCodeModifier.cpp"
  dos2unix $TEST_3RD_PATCH_PATH/*.patch
fi

set -e
readonly ROOT_PATH=$(dirname $(readlink -f "$0"))
CURRENT_DIR=$(pwd)

cd ${ROOT_PATH}/..
PROJ_DIR=$(pwd)

if [ "${INCREMENTAL}" != "ON" ]; then
    rm -rf ./build ./output
fi
mkdir -p "${PROJ_DIR}/output"
mkdir -p "${PROJ_DIR}/build"

if [ "${BUILD_PYTHON}" != "ON" ]; then
    echo "========= [warning] cannot skip build python for meta HA ============"
    BUILD_PYTHON=ON
fi

VERSION="$(cat VERSION | tr -d '[:space:]')"
export MEMCACHE_VERSION="${VERSION}"
echo "VERSION IS ${VERSION}"
GIT_COMMIT=`git rev-parse HEAD` || true
{
  echo "memcache_hybrid version info:"
  echo "memcache_hybrid version: ${MEMCACHE_VERSION}"
  echo "git: ${GIT_COMMIT}"
} > "${PROJ_DIR}/output/VERSION"

cp -f "${PROJ_DIR}/output/VERSION" "${PROJ_DIR}/src/memcache/python/memcache_hybrid/"
rm -f "${PROJ_DIR}/output/VERSION"

readonly BACK_PATH_EVN=$PATH

# 如果 PYTHON_HOME 不存在，则设置默认值
if [ -z "$PYTHON_HOME" ]; then
    # 定义要检查的目录路径
    CHECK_DIR="/usr/local/python3.11"
    # 判断目录是否存在
    if [ -d "$CHECK_DIR" ]; then
        export PYTHON_HOME="$CHECK_DIR"
    else
        export PYTHON_HOME="/usr/local"
    fi
    echo "Not set PYTHON_HOME, and use $PYTHON_HOME"
fi

export PYTHON_HOME="${PYTHON_HOME%/}" # remove the last char "/"

export LD_LIBRARY_PATH=$PYTHON_HOME/lib:$LD_LIBRARY_PATH
export PATH=$PYTHON_HOME/bin:$BACK_PATH_EVN
export CMAKE_PREFIX_PATH="${PYTHON_HOME}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

if command -v ninja &> /dev/null; then
    export GENERATOR="Ninja"
    export MAKE_CMD=ninja
else
    export GENERATOR="Unix Makefiles"
    export MAKE_CMD=make
fi

cmake \
    -G "$GENERATOR" \
    -DCMAKE_BUILD_TYPE="${BUILD_MODE}" \
    -DBUILD_TESTS="${BUILD_TESTS}" \
    -DBUILD_OPEN_ABI="${BUILD_OPEN_ABI}" \
    -DBUILD_PYTHON="${BUILD_PYTHON}" \
    -DENABLE_PTRACER="${ENABLE_PTRACER}" \
    -DBUILD_UBSIO="${BUILD_UBSIO}" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    "${MMC_OPTION_ARGS[@]}" \
    -S . -B build/

${MAKE_CMD} install -j"${MMC_BUILD_JOBS}" -C build/

mkdir -p "${PROJ_DIR}/src/memcache/python/memcache_hybrid/lib"
\cp -v "${PROJ_DIR}/output/memcache/lib64/libmf_memcache.so" "${PROJ_DIR}/src/memcache/python/memcache_hybrid/lib"
if [ "${BUILD_UBSIO:-OFF}" == "ON" ]; then
    \cp -d -v "${PROJ_DIR}/output/3rdparty/ubsio/lib/"*.so* "${PROJ_DIR}/src/memcache/python/memcache_hybrid/lib" 2>/dev/null || true
fi
mkdir -p "${PROJ_DIR}/src/memcache/python/memcache_hybrid/config"
\cp -v "${PROJ_DIR}"/config/* "${PROJ_DIR}/src/memcache/python/memcache_hybrid/config"

cd "${PROJ_DIR}"
rm -f "${PROJ_DIR}"/src/memcache/python/memcache_hybrid/_pymmc.cpython*.so
\cp -v "${PROJ_DIR}"/build/src/memcache/csrc/python_wrapper/_pymmc.cpython*.so "${PROJ_DIR}"/src/memcache/python/memcache_hybrid
cd "${PROJ_DIR}/src/memcache/python"
rm -rf build memcache_hybrid.egg-info
export LD_LIBRARY_PATH="${PROJ_DIR}/src/memcache/python/memcache_hybrid/lib":$LD_LIBRARY_PATH # fix `auditwheel repair` failed
if [ "${BUILD_UBSIO:-OFF}" == "ON" ]; then
    export LD_LIBRARY_PATH="${PROJ_DIR}/output/3rdparty/ubsio/lib":$LD_LIBRARY_PATH # fix `auditwheel repair` failed
fi
python3 setup.py bdist_wheel

mkdir -p "${PROJ_DIR}/output/memcache/wheel"
cp "${PROJ_DIR}"/src/memcache/python/dist/*.whl "${PROJ_DIR}/output/memcache/wheel"
rm -rf "${PROJ_DIR}"/src/memcache/python/dist
rm -rf "${PROJ_DIR}/src/memcache/python/memcache_hybrid/VERSION"
rm -rf "${PROJ_DIR}/src/memcache/python/memcache_hybrid/config"

cd ${CURRENT_DIR}
