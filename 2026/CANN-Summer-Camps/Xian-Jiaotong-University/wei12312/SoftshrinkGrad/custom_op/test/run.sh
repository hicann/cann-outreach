#!/bin/bash
set -euo pipefail

: "${ASCEND_TOOLKIT_HOME:?Please set ASCEND_TOOLKIT_HOME first.}"
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"

export DDK_PATH="${ASCEND_TOOLKIT_HOME}"
export NPU_HOST_LIB="${ASCEND_TOOLKIT_HOME}/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64"
: "${CUSTOM_OPP_PATH:=${HOME}/vendors/customize}"
export CUSTOM_OPP_PATH

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
BUILD_DIR="${SCRIPT_DIR}/build"
cmake -E remove_directory "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_SKIP_RPATH=TRUE
cmake --build "${BUILD_DIR}" -j"$(nproc)"

export LD_LIBRARY_PATH="${CUSTOM_OPP_PATH}/op_api/lib:${LD_LIBRARY_PATH:-}"
"${BUILD_DIR}/execute_softshrink_grad"
