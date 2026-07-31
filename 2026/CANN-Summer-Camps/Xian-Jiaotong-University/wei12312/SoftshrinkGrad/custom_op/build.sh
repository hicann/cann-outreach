#!/bin/bash
set -euo pipefail

: "${ASCEND_TOOLKIT_HOME:?Please set ASCEND_TOOLKIT_HOME first.}"

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
BUILD_DIR="${SCRIPT_DIR}/build_out"

source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
cmake -E remove_directory "${BUILD_DIR}"
cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" --preset=default
cmake --build "${BUILD_DIR}" --target binary -j"$(nproc)"
cmake --build "${BUILD_DIR}" --target "${1:-package}" -j"$(nproc)"
