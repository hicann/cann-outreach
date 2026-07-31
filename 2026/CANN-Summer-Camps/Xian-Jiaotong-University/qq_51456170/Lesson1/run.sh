#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    if [ -n "${ASCEND_HOME_PATH:-}" ]; then
        export ASCEND_TOOLKIT_HOME="${ASCEND_HOME_PATH}"
    elif [ -d "/home/developer/Ascend/cann-9.0.0" ]; then
        export ASCEND_TOOLKIT_HOME="/home/developer/Ascend/cann-9.0.0"
    else
        echo "Error: ASCEND_TOOLKIT_HOME is not set."
        exit 1
    fi
fi

source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
export ASC_DIR="${ASCEND_HOME_PATH}/aarch64-linux/tikcpp/ascendc_kernel_cmake/"

cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build"
cmake --build "${SCRIPT_DIR}/build" --target mul_test -j"$(nproc)"
"${SCRIPT_DIR}/build/mul_test"
