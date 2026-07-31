#!/usr/bin/env bash
set -euo pipefail

: "${ASCEND_TOOLKIT_HOME:?ASCEND_TOOLKIT_HOME is not set}"
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set}"

source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
export ASC_DIR="${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/"

cmake -S . -B build -DASC_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-2201}"
cmake --build build
./build/mul_test
