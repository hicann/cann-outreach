#!/bin/bash
set -e

source "$ASCEND_TOOLKIT_HOME/set_env.sh"
mkdir -p build
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
cd build/
cmake ..
make
./mul_test
