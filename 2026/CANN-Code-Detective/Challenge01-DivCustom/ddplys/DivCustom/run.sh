#!/bin/bash
set -euo pipefail

# 激活cann环境，可根据实际情况修改，在线环境一般不用修改，
: "${ASCEND_TOOLKIT_HOME:?ASCEND_TOOLKIT_HOME is not set}"
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set by set_env.sh}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

mkdir -p "$BUILD_DIR"
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR"
"$BUILD_DIR/div_test"
