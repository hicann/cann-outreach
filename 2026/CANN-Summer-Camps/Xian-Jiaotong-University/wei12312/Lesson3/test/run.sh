#!/bin/bash
set -euo pipefail

if [[ -n "${ASCEND_INSTALL_PATH:-}" ]]; then
    _ASCEND_INSTALL_PATH="${ASCEND_INSTALL_PATH}"
elif [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    _ASCEND_INSTALL_PATH="${ASCEND_HOME_PATH}"
else
    if [[ -d "${HOME}/Ascend/ascend-toolkit/latest" ]]; then
        _ASCEND_INSTALL_PATH="${HOME}/Ascend/ascend-toolkit/latest"
    else
        _ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
    fi
fi

source "${_ASCEND_INSTALL_PATH}/bin/setenv.bash"
export DDK_PATH="${_ASCEND_INSTALL_PATH}"
export NPU_HOST_LIB="${_ASCEND_INSTALL_PATH}/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64"

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
BUILD_DIR="${SCRIPT_DIR}/build"
cmake -E remove_directory "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DCMAKE_SKIP_RPATH=TRUE
cmake --build "${BUILD_DIR}" -j"$(nproc)"
(
    cd "${BUILD_DIR}"
    export LD_LIBRARY_PATH="${_ASCEND_INSTALL_PATH}/opp/vendors/customize/op_api/lib:${LD_LIBRARY_PATH:-}"
    ./execute_add_op
)
