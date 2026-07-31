#!/bin/bash
set -eu

source "$ASCEND_TOOLKIT_HOME/set_env.sh"
mkdir -p build
ARCH="$(uname -m)"
export ASC_DIR="$ASCEND_HOME_PATH/$ARCH-linux/tikcpp/ascendc_kernel_cmake/"
cd build/
cmake ..
make
./mul_test
