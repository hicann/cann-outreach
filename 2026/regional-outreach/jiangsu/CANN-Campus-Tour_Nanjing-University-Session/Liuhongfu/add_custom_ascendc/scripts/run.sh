#!/bin/bash
# =============================================================================
# run.sh —— 运行 add_custom 样例并校验精度
#
# 用法:
#   ./scripts/run.sh [deviceId]
#   ./scripts/run.sh 0
# =============================================================================
set -e

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_OUT="${CUR_DIR}/build_test"
DEVICE_ID="${1:-0}"

if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH 未设置，请先 source set_env.sh"
    exit 1
fi
if [ ! -d "${TEST_OUT}" ]; then
    echo "[ERROR] 未找到 ${TEST_OUT}，请先执行 ./scripts/build.sh"
    exit 1
fi

# 让运行时能找到自定义算子包
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}/vendors/customize:${ASCEND_CUSTOM_OPP_PATH}"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}/vendors/customize/op_api/lib:${LD_LIBRARY_PATH}"

FAILED=0

# 覆盖几组典型 shape：
#   [8, 257]  -> 列数非 32B 对齐，且元素总数不能被核数整除，命中尾块 + 空分片分支
#   [1, 16]   -> 小张量，只有一个 tile
#   [64, 64]  -> 完全对齐的常规情况
#   [3, 1000] -> 奇数行、非对齐列
for SHAPE in "8 257" "1 16" "64 64" "3 1000"; do
    set -- ${SHAPE}
    echo ""
    echo "############################################################"
    echo "# test_aclnn_add_custom  N2=$1 N1=$2"
    echo "############################################################"
    if ! "${TEST_OUT}/test_aclnn_add_custom" "${DEVICE_ID}" "$1" "$2"; then
        FAILED=1
    fi
done

if [ -x "${TEST_OUT}/test_ge_graph_add_custom" ]; then
    echo ""
    echo "############################################################"
    echo "# test_ge_graph_add_custom  N2=8 N1=257"
    echo "############################################################"
    if ! "${TEST_OUT}/test_ge_graph_add_custom" 8 257; then
        FAILED=1
    fi
fi

echo ""
if [ "${FAILED}" -eq 0 ]; then
    echo "[SUCCESS] 全部用例通过"
else
    echo "[FAILED] 存在失败用例，请检查上面的输出"
fi
exit "${FAILED}"
