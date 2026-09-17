#!/bin/bash
# ============================================================
# atanh_custom 算子运行验证脚本
# ------------------------------------------------------------
# 功能：一键完成「生成测试数据 → 执行算子 → 精度对比」全流程
#
# 用法：
#   bash run.sh [N4 N3 N2 N1]
#   - 四个参数对应输入 shape [N4, N3, N2, N1]，默认 2 3 4 5
#
# 前置条件：
#   - 已执行 bash build.sh 完成编译
#   - 已正确设置 CANN 环境变量
# ============================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${SCRIPT_DIR}/build
TEST_DIR=${SCRIPT_DIR}/testcase

# ---------- 1. 设置 CANN 环境 ----------
if [ -z "${ASCEND_CANN_PACKAGE_PATH}" ]; then
    ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
    export ASCEND_CANN_PACKAGE_PATH
fi
if [ -f "${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash" ]; then
    source "${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash"
elif [ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]; then
    source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
fi

# ---------- 2. 解析 shape ----------
N4=${1:-2}
N3=${2:-3}
N2=${3:-4}
N1=${4:-5}
export ATANH_SHAPE="${N4},${N3},${N2},${N1}"

echo "============================================"
echo ">>> atanh_custom 运行验证"
echo "    shape = [${N4}, ${N3}, ${N2}, ${N1}]"
echo "============================================"

# 统一在工程根目录下生成/读取数据文件
cd "${SCRIPT_DIR}"

# ---------- 3. 生成测试数据（input.bin + 真值 golden.bin + shape.txt） ----------
echo ""
echo ">>> 生成测试数据 ..."
python3 "${TEST_DIR}/gen_data.py"

# ---------- 4. 执行算子 ----------
echo ""
echo ">>> 执行算子 ..."
"${BUILD_DIR}/main"

# ---------- 5. 精度对比 ----------
echo ""
echo ">>> 精度对比 ..."
python3 "${TEST_DIR}/verify_result.py"

echo ""
echo "============================================"
echo ">>> 运行验证完成"
echo "============================================"
