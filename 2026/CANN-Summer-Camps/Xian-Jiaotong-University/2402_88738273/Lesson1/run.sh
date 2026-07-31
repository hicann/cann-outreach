#!/bin/bash
set -euo pipefail

if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    echo "Error: ASCEND_TOOLKIT_HOME is not set"
    exit 1
fi
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
export ASC_DIR="${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/"

mkdir -p build/
cd build/ && \
cmake .. && \
make -j"$(nproc)" && \
./mul_test
