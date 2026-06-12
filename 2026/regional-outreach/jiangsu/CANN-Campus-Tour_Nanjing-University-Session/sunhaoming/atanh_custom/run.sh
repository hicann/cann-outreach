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

# ============================================================================
# atanh 算子 Kernel 直调运行脚本
# ============================================================================
#
# 编译 → 生成测试数据 → 运行 → 精度验证
#
# 用法：
#   bash run.sh                            # 默认 shape [1,1,1,128]
#   bash run.sh --shape=1,1,4,2048         # 指定 4D shape
#   bash run.sh --skip-build                # 跳过编译
#   bash run.sh --all                       # 运行多个 shape
#
# 退出码：
#   0  全部步骤成功
#   1  某步骤失败
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="atanh_custom"
SHAPE="1,1,1,128"
SKIP_BUILD=0
RUN_ALL=0

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --all)        RUN_ALL=1 ;;
        --shape=*)    SHAPE="${arg#*=}" ;;
        --help|-h)
            echo "Usage: bash run.sh [--shape=N4,N3,N2,N1] [--skip-build] [--all]"
            exit 0
            ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置，请先 source set_env.sh"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 执行失败"

# 多个 shape 测试
if [ "${RUN_ALL}" -eq 1 ]; then
    echo "=== 运行全部预设测试 ==="
    for s in "1,1,1,128" "1,1,4,2048" "1,4,8,4096" "2,4,8,4096"; do
        echo ""
        echo ">>> Testing shape=[${s}] <<<"
        bash "$0" --shape="$s"
        echo ""
    done
    echo "=== 全部测试完成 ==="
    exit 0
fi

# 解析 shape 计算总元素数
IFS=',' read -ra DIMS <<< "$SHAPE"
TOTAL=1
for d in "${DIMS[@]}"; do TOTAL=$((TOTAL * d)); done

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 指定但 build/${OP_NAME} 不存在"
    echo "=== [2/4] 跳过编译（复用已有产物）==="
else
    echo "=== [2/4] 编译 ==="
    mkdir -p build
    cd build
    cmake .. 2>&1 || die "cmake 配置失败"
    make -j$(nproc) 2>&1 || die "make 编译失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据 (shape=[${SHAPE}], total=${TOTAL}) ==="
cd build
python3 ../scripts/gen_data.py "$SHAPE" || die "gen_data.py 执行失败"

echo "=== [4/4] 运行 Kernel ==="
rm -f output/output.bin
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
"./${OP_NAME}" "$SHAPE" || die "Kernel 运行失败（exit code $?）"
[ -f output/output.bin ] || die "Kernel 运行后 output.bin 不存在"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "精度验证失败"

echo "=== 完成 (shape=[${SHAPE}]) ==="
exit 0
