#!/bin/bash
#==============================================================================
# build.sh - Build the abs_ascend Ascend C operator
#
# Prerequisites:
#   - Ascend CANN toolkit installed (CANN >= 7.0)
#   - ASCEND_HOME or ASCEND_TOOLKIT_HOME environment variable set
#
# Usage:
#   bash build.sh [Debug|Release]
#==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE="${1:-Release}"
BUILD_DIR="${SCRIPT_DIR}/build"

# Check Ascend environment
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

echo "=== abs_ascend Build ==="
echo "  Build type:  ${BUILD_TYPE}"
echo "  Toolkit:     ${ASCEND_TOOLKIT_HOME}"
echo "  Output dir:  ${BUILD_DIR}"
echo ""

# Create build directory
mkdir -p "${BUILD_DIR}"

# Configure
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DASCEND_C_COMPILER="$(which cce)" \
    -DCMAKE_INSTALL_PREFIX="${BUILD_DIR}/output"

# Build
cmake --build "${BUILD_DIR}" --target install -j"$(nproc)"

echo ""
echo "=== Build complete ==="
echo "Operator package: ${BUILD_DIR}/output/"
echo ""
echo "To run tests:"
echo "  python3 ${SCRIPT_DIR}/scripts/test_abs.py --op-dir ${BUILD_DIR}/output"
