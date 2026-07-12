#!/bin/bash
set -e
if [ -z "$BASE_LIBS_PATH" ]; then
    if [ -z "$ASCEND_HOME_PATH" ]; then
        if [ -z "$ASCEND_AICPU_PATH" ]; then
            echo "please set env."
            exit 1
        else
            export ASCEND_HOME_PATH="$ASCEND_AICPU_PATH"
        fi
    else
        export ASCEND_HOME_PATH="$ASCEND_HOME_PATH"
    fi
else
    export ASCEND_HOME_PATH="$BASE_LIBS_PATH"
fi
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$ASCEND_HOME_PATH}"
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path=$(realpath "$(dirname "$0")")

BUILD_DIR="build_out"
mkdir -p build_out
rm -rf build_out/*

cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

target=package
if [ "${1:-}"x != ""x ]; then target="$1"; fi

if [ "$cmake_version" \< "3.19.0" ] ; then
    read -r -a opts <<< "$(python3 "$ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py" "$script_path/CMakePresets.json")"
    cmake -S . -B "$BUILD_DIR" "${opts[@]}"
else
    cmake -S . -B "$BUILD_DIR" --preset=default
fi
cmake --build "$BUILD_DIR" --target binary -j$(nproc)
cmake --build "$BUILD_DIR" --target "$target" -j$(nproc)
