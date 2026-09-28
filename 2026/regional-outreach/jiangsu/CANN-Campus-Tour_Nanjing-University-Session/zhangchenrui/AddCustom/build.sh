#!/bin/bash
set -e

# 设置 CANN 环境变量（根据实际安装路径修改）
export ASCEND_HOME=/usr/local/Ascend
export PATH=$ASCEND_HOME/ascend-toolkit/latest/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH

mkdir -p build
cd build
cmake ..
make -j