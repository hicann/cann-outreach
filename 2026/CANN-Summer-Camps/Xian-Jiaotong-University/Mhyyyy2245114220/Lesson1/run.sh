#!/bin/bash
# 编译并运行 mul_custom 算子
set -e

# 1. 加载 CANN 环境 (请按实际安装路径调整)
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

BUILD_DIR=build
if [ ! -d "$BUILD_DIR" ]; then
    mkdir "$BUILD_DIR"
fi
cd "$BUILD_DIR"

# 2. CMake 配置 + 编译
cmake ..
make -j$(nproc)

# 3. 运行
echo "==== Running mul_custom ===="
./mul_custom
