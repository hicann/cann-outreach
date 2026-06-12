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
# Ascend C Kernel 直调运行脚本 - Gcd 算子（FP16, 4D Broadcast）
# ============================================================================
#
# 用法:
#   bash run.sh [preset|dims...]
#
# 预设用例:
#   bash run.sh            # 默认: A (同 shape)
#   bash run.sh A          # self=[2,4,8,16] other=[2,4,8,16]
#   bash run.sh B          # self=[1,4,8,16] other=[2,4,8,16] — self broadcast
#   bash run.sh C          # self=[2,1,8,1]  other=[2,4,8,16] — 双 broadcast
#   bash run.sh D          # self=[1,1,1,1]  other=[2,4,8,16] — 全 broadcast
#
# 自定义 4D shape:
#   bash run.sh <N4> <N3> <N2> <N1> <M4> <M3> <M2> <M1>
#   bash run.sh --skip-build
#   bash run.sh --torch
#
# 退出码: 0=成功, 1=失败
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="gcd_custom"
SKIP_BUILD=0
TORCH_ONLY=0
SHAPE_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
        --help|-h)
            echo "Usage: bash run.sh [OPTIONS] [preset|dims...]"
            echo "Presets: A (same shape), B (self broadcast), C (dual broadcast), D (full broadcast)"
            echo "Custom:  <N4> <N3> <N2> <N1> <M4> <M3> <M2> <M1>"
            echo "Options: --skip-build, --torch"
            exit 0
            ;;
        *) SHAPE_ARGS+=("$arg") ;;
    esac
done

die() { echo "ERROR: $*" >&2; exit 1; }

echo "=== [1/4] 设置 CANN 环境 ==="
[ -n "${ASCEND_HOME_PATH:-}" ] || die "ASCEND_HOME_PATH 未设置"
source "${ASCEND_HOME_PATH}/set_env.sh" || die "set_env.sh 执行失败"

if [ "${TORCH_ONLY}" -eq 1 ]; then
    echo "=== PyTorch 通路验证 ==="
    [ -f "build/${OP_NAME}" ] || die "请先编译: cmake .. && make"
    cd build
    python3 ../scripts/test_torch.py "${SHAPE_ARGS[@]}" || die "PyTorch 验证失败"
    echo "=== 完成 ==="; exit 0
fi

if [ "${SKIP_BUILD}" -eq 1 ]; then
    [ -f "build/${OP_NAME}" ] || die "--skip-build 但 build/${OP_NAME} 不存在"
    echo "=== [2/4] 跳过编译 ==="
else
    echo "=== [2/4] 编译 ==="
    mkdir -p build && cd build && cmake .. && make -j4 && cd ..
fi

echo "=== [3/4] 生成测试数据 ==="
cd build
python3 ../scripts/gen_data.py "${SHAPE_ARGS[@]}" || die "gen_data.py 失败"

echo "=== [4/4] 运行 Kernel ==="
rm -f output/output.bin
"./${OP_NAME}" "${SHAPE_ARGS[@]}" || die "Kernel 运行失败"
[ -f output/output.bin ] || die "output.bin 不存在"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin || die "精度验证失败"

echo "=== PyTorch 通路验证 ==="
python3 ../scripts/test_torch.py "${SHAPE_ARGS[@]}" || die "PyTorch 验证失败"

echo "=== 完成 ==="
exit 0
