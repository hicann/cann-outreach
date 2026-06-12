#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
BUILD_DIR="$PROJECT_ROOT/build"

echo "=========================================="
echo "  relu_custom 算子编译 & 运行"
echo "  4D shape, float16, ND Format"
echo "=========================================="

if [ -z "$ASCEND_HOME_PATH" ]; then
    if [ -d "$HOME/Ascend/cann" ]; then
        export ASCEND_HOME_PATH="$HOME/Ascend/cann"
    elif [ -d "/usr/local/Ascend/cann" ]; then
        export ASCEND_HOME_PATH="/usr/local/Ascend/cann"
    else
        echo "ERROR: ASCEND_HOME_PATH 未设置，找不到 CANN 安装路径"
        exit 1
    fi
fi
echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"

echo ""
echo "[1/2] 编译..."
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ..
make -j$(nproc)
echo "编译完成"

echo ""
echo "[2/2] 运行测试..."
echo ""
./relu_custom

echo ""
echo "完成"
