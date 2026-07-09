#!/usr/bin/env bash
set -euo pipefail

source "${ASCEND_TOOLKIT_HOME}/set_env.sh"

export ASC_DIR="${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/"

rm -rf build
mkdir -p build
cd build

cmake ..
make -j"$(nproc)"

./mul_test