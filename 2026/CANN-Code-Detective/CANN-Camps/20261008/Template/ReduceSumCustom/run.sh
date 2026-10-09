#!/bin/bash
# 激活cann环境，可根据实际情况修改，在线环境一般不用修改
source "$ASCEND_TOOLKIT_HOME/set_env.sh"

# 获取当前脚本所在目录，确保在任何路径下执行都能找到正确文件
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

mkdir -p build
cd build/ && \
cmake .. && \
make -j && \
./reduce_sum_custom
