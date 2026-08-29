#!/bin/bash
set -euo pipefail

if [ -z "${BASE_LIBS_PATH:-}" ]; then
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        if [ -z "${ASCEND_AICPU_PATH:-}" ]; then
            echo "please set env."
            exit 1
        else
            export ASCEND_HOME_PATH=$ASCEND_AICPU_PATH
        fi
    fi
else
    export ASCEND_HOME_PATH=$BASE_LIBS_PATH
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"

# 切换到脚本所在目录，避免 rm/cmake 依赖调用时的工作目录
script_path=$(realpath "$(dirname "$0")")
cd "$script_path"

BUILD_DIR="build_out"
mkdir -p build_out
rm -rf build_out/*

target=package
if [ "$#" -ge 1 ]; then target=$1; fi

# binaryDir 已由 CMakePresets.json 的 preset 指定，无需重复传 -B
cmake -S . --preset=default
cmake --build "$BUILD_DIR" --target binary -j"$(nproc)"
cmake --build "$BUILD_DIR" --target "$target" -j"$(nproc)"
