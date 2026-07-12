#!/usr/bin/env bash
set -eo pipefail

if [[ -n "${BASE_LIBS_PATH:-}" ]]; then
    ASCEND_HOME_PATH="$BASE_LIBS_PATH"
elif [[ -z "${ASCEND_HOME_PATH:-}" && -n "${ASCEND_AICPU_PATH:-}" ]]; then
    ASCEND_HOME_PATH="$ASCEND_AICPU_PATH"
fi

: "${ASCEND_HOME_PATH:?Please set the CANN environment before building}"
export ASCEND_HOME_PATH

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build_out"
rm -rf "$BUILD_DIR"

if cmake --help | grep -q -- '--preset'; then
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" --preset=default \
        -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
else
    PRESET_PARSE="$ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py"
    CMAKE_OPTIONS_TEXT="$(python3 "$PRESET_PARSE" "$SCRIPT_DIR/CMakePresets.json")"
    read -r -a CMAKE_OPTIONS <<< "$CMAKE_OPTIONS_TEXT"
    cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" "${CMAKE_OPTIONS[@]}" \
        -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
fi

cmake --build "$BUILD_DIR" --target binary package -j"$(nproc)"
