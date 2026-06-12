#!/bin/bash
# build.sh - Build the softmax_ascend Ascend C operator
#
# Prerequisites:
#   - Ascend CANN toolkit installed (CANN >= 7.0)
#   - ASCEND_HOME or ASCEND_TOOLKIT_HOME environment variable set
#
# Usage:
#   bash build.sh [Debug|Release]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE="${1:-Release}"
BUILD_DIR="${SCRIPT_DIR}/build"

if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    if [ -z "${ASCEND_HOME:-}" ]; then
        echo "ERROR: ASCEND_TOOLKIT_HOME or ASCEND_HOME not set."
        echo "Please source the Ascend environment first:"
        echo "  source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        exit 1
    else
        export ASCEND_TOOLKIT_HOME="${ASCEND_HOME}"
    fi
fi

echo "=== softmax_ascend Build ==="
echo "  Build type:  ${BUILD_TYPE}"
echo "  Toolkit:     ${ASCEND_TOOLKIT_HOME}"
echo "  Output dir:  ${BUILD_DIR}"
echo ""

mkdir -p "${BUILD_DIR}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DASCEND_C_COMPILER="$(which cce)" \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/output"

cmake --build "${BUILD_DIR}" --target install -j"$(nproc)"

echo ""
echo "=== Build complete ==="
echo "Operator package: ${BUILD_DIR}/output/"
echo ""
echo "To run tests:"
echo "  python3 ${SCRIPT_DIR}/scripts/test_softmax.py --op-dir ${BUILD_DIR}/output"
