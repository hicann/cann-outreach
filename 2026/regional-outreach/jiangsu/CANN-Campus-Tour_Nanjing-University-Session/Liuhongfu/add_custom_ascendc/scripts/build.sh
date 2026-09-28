#!/bin/bash
# =============================================================================
# build.sh
#   1) 生成/复用算子工程骨架
#   2) 编译并安装自定义算子包
#   3) 编译 test/ 下的两个样例程序
#
# 用法:
#   ./scripts/build.sh [SOC_VERSION]
#   环境变量:
#   SKIP_GEN=1    跳过 msopgen，直接编译已有 build_gen 工程
#   SKIP_TEST=1   只编译算子包，不编译样例
# =============================================================================
set -e

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GEN_DIR="${CUR_DIR}/build_gen"
OP_NAME="AddCustom"
SOC_VERSION="${1:-${SOC_VERSION:-Ascend910B2}}"
TEST_OUT="${CUR_DIR}/build_test"

if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH 未设置，请先 source \${ASCEND_HOME_PATH}/set_env.sh"
    exit 1
fi
echo "[INFO] ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}"

# ---------------------------------------------------------------------------
# 1. 工程骨架
# ---------------------------------------------------------------------------
if [ "${SKIP_GEN}" != "1" ] || [ ! -d "${GEN_DIR}/${OP_NAME}" ]; then
    "${CUR_DIR}/scripts/gen_project.sh" "${SOC_VERSION}"
fi
TARGET="${GEN_DIR}/${OP_NAME}"

# ---------------------------------------------------------------------------
# 2. 编译安装算子包
# ---------------------------------------------------------------------------
cd "${TARGET}"
chmod +x ./build.sh 2>/dev/null || true
./build.sh -c "${SOC_VERSION}"
cd -

OPP_VENDOR_DIR="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}/vendors/customize"
echo "[INFO] 自定义算子包目录: ${OPP_VENDOR_DIR}"

# ---------------------------------------------------------------------------
# 3. 编译测试程序
# ---------------------------------------------------------------------------
if [ "${SKIP_TEST}" = "1" ]; then
    echo "[INFO] SKIP_TEST=1，跳过样例编译"
    exit 0
fi

rm -rf "${TEST_OUT}"
mkdir -p "${TEST_OUT}"

ACL_INC="${ASCEND_HOME_PATH}/include"
OP_API_INC="${OPP_VENDOR_DIR}/op_api/include"
OP_PROTO_INC="${OPP_VENDOR_DIR}/op_proto/inc"
LIBS="-L${ASCEND_HOME_PATH}/lib64 -L${OPP_VENDOR_DIR}/op_api/lib"

echo "[INFO] 编译 test_aclnn_add_custom ..."
g++ -std=c++17 -O2 -Wall \
    "${CUR_DIR}/test/test_aclnn_add_custom.cpp" \
    -o "${TEST_OUT}/test_aclnn_add_custom" \
    -I"${ACL_INC}" -I"${OP_API_INC}" -I"${CUR_DIR}/test" \
    ${LIBS} \
    -lascendcl -lnnopbase -lcust_opapi -lpthread
echo "[OK] ${TEST_OUT}/test_aclnn_add_custom"

if [ -d "${OP_PROTO_INC}" ]; then
    echo "[INFO] 编译 test_ge_graph_add_custom ..."
    g++ -std=c++17 -O2 -Wall \
        "${CUR_DIR}/test/test_ge_graph_add_custom.cpp" \
        -o "${TEST_OUT}/test_ge_graph_add_custom" \
        -I"${ACL_INC}" -I"${OP_PROTO_INC}" -I"${CUR_DIR}/test" \
        ${LIBS} \
        -lascendcl -lnnopbase -lgraph -lge_runner -lpthread && \
        echo "[OK] ${TEST_OUT}/test_ge_graph_add_custom"
else
    echo "[WARN] 未找到算子原型头文件目录 ${OP_PROTO_INC}，跳过 GE 图模式样例编译"
fi

echo ""
echo "[DONE] 接下来执行: ${CUR_DIR}/scripts/run.sh"
