#!/bin/bash
# 任一命令失败立即退出，避免环境异常时错误信息混乱难以排查
set -euo pipefail

# 激活cann环境，可根据实际情况修改，在线环境一般不用修改，
: "${ASCEND_TOOLKIT_HOME:?ASCEND_TOOLKIT_HOME is not set}"
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
mkdir -p build
: "${ASCEND_HOME_PATH:?ASCEND_HOME_PATH is not set}"
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
cd build/
cmake ..
make
./div_test
