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

set -e
readonly ROOT_PATH=$(dirname $(readlink -f "$0"))
CURRENT_DIR=$(pwd)

BUILD_MODE="RELEASE"
BUILD_PYTHON="ON"
BUILD_TEST="OFF"
INCREMENTAL="OFF"
BUILD_UBSIO="ON"
BUILD_OPEN_ABI="OFF"
BUILD_PTRACER="ON"
BUILD_KV_EVENTS=""
BUILD_GIT_COMMIT=""
BUILD_OFFLINE=""

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --build_mode <mode>     Set build mode (RELEASE/DEBUG/ASAN), default: RELEASE"
    echo "  --build_test <ON/OFF>   Enable/disable package test utilities, default: OFF"
    echo "  --incremental           Enable incremental build (skip clean), default: OFF"
    echo "  --build_ubsio <ON/OFF>  Enable/disable build and package ubs-io (SSD backend), default: ON"
    echo "  --build_open_abi <ON/OFF>  Optional native C++ ABI, default: OFF"
    echo "  --enable_ptracer <ON/OFF>  Optional trace instrumentation, default: ON"
    echo "  --enable_kv_events <ON/OFF> Preserve the existing CMake default/cache unless specified"
    echo "  --build_git_commit <ON/OFF> Preserve the existing CMake default/cache unless specified"
    echo "  --offline <ON/OFF>       Preserve the existing CMake default/cache unless specified"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --build_mode DEBUG --incremental"
    echo ""
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --build_mode)
            BUILD_MODE="$2"
            shift 2
            ;;
        --build_test)
            BUILD_TEST="$2"
            shift 2
            ;;
        --incremental)
            INCREMENTAL="ON"
            shift 1
            ;;
        --build_ubsio)
            BUILD_UBSIO="$2"
            shift 2
            ;;
        --build_open_abi)
            if [[ "${2:-}" != "ON" && "${2:-}" != "OFF" ]]; then
                echo "build_open_abi requires ON or OFF" >&2; exit 2
            fi
            BUILD_OPEN_ABI="$2"
            shift 2
            ;;
        --enable_ptracer)
            if [[ "${2:-}" != "ON" && "${2:-}" != "OFF" ]]; then
                echo "enable_ptracer requires ON or OFF" >&2; exit 2
            fi
            BUILD_PTRACER="$2"
            shift 2
            ;;
        --enable_kv_events)
            if [[ "${2:-}" != "ON" && "${2:-}" != "OFF" ]]; then
                echo "enable_kv_events requires ON or OFF" >&2; exit 2
            fi
            BUILD_KV_EVENTS="$2"
            shift 2
            ;;
        --build_git_commit)
            if [[ "${2:-}" != "ON" && "${2:-}" != "OFF" ]]; then
                echo "build_git_commit requires ON or OFF" >&2; exit 2
            fi
            BUILD_GIT_COMMIT="$2"
            shift 2
            ;;
        --offline)
            if [[ "${2:-}" != "ON" && "${2:-}" != "OFF" ]]; then
                echo "offline requires ON or OFF" >&2; exit 2
            fi
            BUILD_OFFLINE="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
done

echo "BUILD_MODE: $BUILD_MODE"
echo "BUILD_PYTHON: $BUILD_PYTHON"
echo "BUILD_UBSIO: $BUILD_UBSIO"

cd "${ROOT_PATH}"

bash build.sh "${BUILD_MODE}" OFF "${BUILD_OPEN_ABI}" "${BUILD_PYTHON}" "${BUILD_PTRACER}" \
    "${INCREMENTAL}" "${BUILD_UBSIO}" "${BUILD_KV_EVENTS}" "${BUILD_GIT_COMMIT}" "${BUILD_OFFLINE}"

bash run_pkg_maker/make_run.sh "${BUILD_TEST}" "${BUILD_UBSIO}"

cd "${CURRENT_DIR}"
