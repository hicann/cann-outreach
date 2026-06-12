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
# Ascend C Kernel 直调运行脚本 - Atanh 算子（float16）
# ============================================================================
#
# 用法：
#   bash run.sh                           # 默认 shape [1,4,16,64]=4096
#   bash run.sh --shape=0                 # [1,4,16,64]=4096
#   bash run.sh --shape=1                 # [2,8,32,128]=65536
#   bash run.sh --shape=2                 # [4,16,64,256]=1M
#   bash run.sh --dims "4 8 16 32"        # 自定义 4D shape [4,8,16,32]
#   bash run.sh --skip-build              # 跳过编译
#   bash run.sh --torch                   # 只跑 PyTorch 通路
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OP_NAME="atanh_custom"

SKIP_BUILD=0
TORCH_ONLY=0
SHAPE_IDX=0
DIMS=""
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --torch)      TORCH_ONLY=1 ;;
        --shape=*)    SHAPE_IDX="${arg#*=}" ;;
        --dims)       DIMS_MODE=1 ;;
    esac
done

# 处理 --dims 参数：--dims "N4 N3 N2 N1"
for arg in "$@"; do
    if [ "$DIMS_MODE" = "1" ] && [[ "$arg" != "--dims" ]] && [[ "$arg" != "$1" ]]; then
        DIMS="$arg"
        break
    fi
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
    [ -f "build/${OP_NAME}" ] || die "--skip-build 指定但 build/${OP_NAME} 不存在，请先完整编译"
    echo "=== [2/4] 跳过编译（复用已有产物）==="
else
    echo "=== [2/4] 编译 ==="
    mkdir -p build
    cd build
    cmake .. || die "cmake 配置失败"
    make -j4  || die "make 编译失败"
    cd ..
fi

if [ -n "$DIMS" ]; then
    echo "=== [3/4] 生成测试数据（自定义 shape=${DIMS}）==="
    cd build
    python3 ../scripts/gen_data.py --dims "${DIMS}" || die "gen_data.py 执行失败"
    echo "=== [4/4] 运行 Kernel（自定义 shape=${DIMS}）==="
    rm -f output/output.bin
    # 将 dims 字符串转换为单独参数
    # shellcheck disable=SC2086
    "./${OP_NAME}" ${DIMS} || die "Kernel 运行失败（exit code $?）"
else
    SHAPE_NAMES=("[1,4,16,64]" "[2,8,32,128]" "[4,16,64,256]")
    echo "=== [3/4] 生成测试数据（shape=${SHAPE_NAMES[$SHAPE_IDX]}）==="
    cd build
    python3 ../scripts/gen_data.py --shape-idx "${SHAPE_IDX}" || die "gen_data.py 执行失败"
    echo "=== [4/4] 运行 Kernel（shape=${SHAPE_NAMES[$SHAPE_IDX]}）==="
    rm -f output/output.bin
    "./${OP_NAME}" "${SHAPE_IDX}" || die "Kernel 运行失败（exit code $?）"
fi

[ -f output/output.bin ] || die "Kernel 运行后 output.bin 不存在（静默失败）"

echo "=== 精度验证 ==="
python3 ../scripts/verify_result.py output/output.bin output/golden.bin \
    || die "精度验证失败（verify_result.py 返回非零）"

echo "=== PyTorch 通路验证 ==="
python3 ../scripts/test_torch.py \
    || die "PyTorch 通路验证失败（test_torch.py 返回非零）"

echo "=== 完成 ==="
exit 0
