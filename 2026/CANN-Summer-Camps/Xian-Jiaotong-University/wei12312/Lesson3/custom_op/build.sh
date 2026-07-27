#!/bin/bash
set -euo pipefail

if [[ -n "${BASE_LIBS_PATH:-}" ]]; then
    export ASCEND_HOME_PATH="${BASE_LIBS_PATH}"
elif [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    export ASCEND_HOME_PATH
elif [[ -n "${ASCEND_AICPU_PATH:-}" ]]; then
    export ASCEND_HOME_PATH="${ASCEND_AICPU_PATH}"
else
    echo "Please set BASE_LIBS_PATH, ASCEND_HOME_PATH, or ASCEND_AICPU_PATH." >&2
    exit 1
fi

echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path="$(realpath "$(dirname "$0")")"

BUILD_DIR="${script_path}/build_out"
cmake -E remove_directory "${BUILD_DIR}"
cmake -E make_directory "${BUILD_DIR}"

target="${1:-package}"

cmake -S "${script_path}" -B "${BUILD_DIR}" --preset=default
cmake --build "${BUILD_DIR}" --target binary -j"$(nproc)"
cmake --build "${BUILD_DIR}" --target "${target}" -j"$(nproc)"
