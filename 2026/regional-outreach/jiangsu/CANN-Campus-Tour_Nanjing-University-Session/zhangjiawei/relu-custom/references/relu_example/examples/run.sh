#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/../build_out"
ASCEND_HOME="${ASCEND_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

g++ -std=c++17 -o "${BUILD_DIR}/test_aclnn_relu" \
    "${SCRIPT_DIR}/test_aclnn_relu.cpp" \
    -I"${ASCEND_HOME}/include" -I"${ASCEND_HOME}/include/third_party" \
    -I"${SCRIPT_DIR}/../op_api" \
    -L"${ASCEND_HOME}/lib64" \
    -lascendc_kernels -lacl_op_api -lacl -lascendc_host -lpthread

echo ""
echo "=== Running relu test ==="
"${BUILD_DIR}/test_aclnn_relu"
