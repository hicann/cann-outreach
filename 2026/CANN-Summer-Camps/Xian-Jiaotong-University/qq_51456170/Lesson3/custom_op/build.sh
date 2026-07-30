#!/bin/bash
set -euo pipefail

if [ -z "${BASE_LIBS_PATH:-}" ]; then
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        if [ -z "${ASCEND_AICPU_PATH:-}" ]; then
            if [ -d "/home/developer/Ascend/cann-9.0.0" ]; then
                export ASCEND_HOME_PATH="/home/developer/Ascend/cann-9.0.0"
            else
                echo "please set env."
                exit 1
            fi
        else
            export ASCEND_HOME_PATH="$ASCEND_AICPU_PATH"
        fi
    else
        export ASCEND_HOME_PATH="$ASCEND_HOME_PATH"
    fi
else
    export ASCEND_HOME_PATH="$BASE_LIBS_PATH"
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path=$(realpath "$(dirname "$0")")
cd "$script_path"

BUILD_DIR="build_out"
mkdir -p "$BUILD_DIR"
rm -rf "$BUILD_DIR"/*


target=package
if [ "${1:-}"x != ""x ]; then target="$1"; fi

cmake -S . -B "$BUILD_DIR" --preset=default \
    -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
    -DASCEND_COMPUTE_UNIT="ascend910b;ascend910_93"
cmake --build "$BUILD_DIR" --target binary -j"$(nproc)"
cmake --build "$BUILD_DIR" --target "$target" -j"$(nproc)"
