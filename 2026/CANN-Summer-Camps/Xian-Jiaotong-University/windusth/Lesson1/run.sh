#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$ASCEND_TOOLKIT_HOME/set_env.sh"
rm -rf build
mkdir -p build

export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
cmake -S . -B build
cmake --build build -j"$(nproc)"
./build/mul_test
