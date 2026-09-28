#!/bin/bash
# Build entry for the Gcd123 custom operator.
#
# Prerequisites (Linux machine with a CANN toolkit installed):
#   export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
#   source ${ASCEND_TOOLKIT_HOME}/bin/setenv.bash
#
# Usage:
#   ./build.sh <soc_version>            # e.g. ./build.sh ascend910b4
#
# Artifacts:
#   build_out/custom_opp_<soc>_<os>_<arch>.run  -- install with:
#   ./build_out/custom_opp_*.run
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_out"
SOC_VERSION="${1:-ascend910b4}"

if [ -z "${ASCEND_TOOLKIT_HOME}" ]; then
    echo "ERROR: ASCEND_TOOLKIT_HOME is not set. Source CANN setenv.bash first."
    exit 1
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SCRIPT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DSOC_VERSION="${SOC_VERSION}" \
    -DCMAKE_SKIP_RPATH=TRUE

make -j"$(nproc)" binary package

echo ""
echo "Build finished. Install package:"
ls -1 "${BUILD_DIR}"/custom_opp_*.run
echo "Install with: ./build_out/custom_opp_*.run"
