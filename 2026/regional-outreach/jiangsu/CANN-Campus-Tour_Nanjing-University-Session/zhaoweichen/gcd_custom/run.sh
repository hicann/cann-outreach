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
# Ascend C Kernel 直调运行脚本 - Gcd 算子 (broadcast)
# ============================================================================
#
# 用法:
#   bash run.sh                      # 场景 1: self=[4,2,1,8] other=[4,1,6,8]
#   bash run.sh [1|2|3]              # 选择测试场景
#   bash run.sh --torch              # 只跑 PyTorch 通路
#   bash run.sh --skip-build         # 跳过编译
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="gcd"
SCENE="1"

SKIP_BUILD=0
TORCH_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
        1|2|3)        SCENE="$arg" ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置，请先配置 CANN 环境"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 执行失败"

if [ "${TORCH_ONLY}" -eq 1 ]; then
    echo "=== PyTorch 通路验证 ==="
    cd build
    python3 ../scripts/test_torch.py || die "PyTorch 通路验证失败"
    echo "=== 完成 ==="
    exit 0
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 指定但 build/${OP_NAME} 不存在"
    echo "=== [2/4] 跳过编译（复用已有产物）==="
else
    echo "=== [2/4] 编译 ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake 配置失败"
    make -j4  || die "make 编译失败"
    cd ..
fi

echo "=== [3/4] 生成测试数据（场景 ${SCENE}）==="
cd build
python3 ../scripts/gen_data.py "${SCENE}" || die "gen_data.py 执行失败"

echo "=== [4/4] 运行 Kernel ==="
rm -f output/output.bin

# 解析场景形状传递给可执行文件
case "${SCENE}" in
    1) "./${OP_NAME}" 4 2 1 8  4 1 6 8  ;;
    2) "./${OP_NAME}" 1 4 4 4  2 4 4 4  ;;
    3) "./${OP_NAME}" 3 1 5 1  1 4 1 6  ;;
esac || die "Kernel 运行失败"

[ -f output/output.bin ] || die "Kernel 运行后 output.bin 不存在（静默失败）"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "精度验证失败"

echo "=== PyTorch 通路验证 ==="
python3 ../scripts/test_torch.py \
    || die "PyTorch 通路验证失败"

echo "=== 完成 ==="
exit 0
