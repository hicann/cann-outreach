#!/bin/bash
set -euo pipefail
if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    echo "错误: ASCEND_TOOLKIT_HOME 未设置"
    exit 1
fi
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
mkdir -p build
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake" && \
cd build/ && \
cmake .. && \
make

./mul_test