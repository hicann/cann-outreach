#!/bin/bash
set -euo pipefail
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

if [ -z "${BASE_LIBS_PATH:-}" ]; then
    if [ -z "${ASCEND_HOME_PATH:-}" ]; then
        if [ -z "${ASCEND_AICPU_PATH:-}" ]; then
            echo "please set env."
            exit 1
        else
            export ASCEND_HOME_PATH="$ASCEND_AICPU_PATH"
        fi
    else
<<<<<<< HEAD
        export ASCEND_HOME_PATH
=======
<<<<<<< HEAD
        export ASCEND_HOME_PATH="$ASCEND_HOME_PATH"
=======
        export ASCEND_HOME_PATH
>>>>>>> lesson3
>>>>>>> lesson1
    fi
else
    export ASCEND_HOME_PATH="$BASE_LIBS_PATH"
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path=$(realpath "$(dirname "$0")")

BUILD_DIR="build_out"
mkdir -p "$BUILD_DIR"
rm -rf "$BUILD_DIR"/*
opts=$(python3 "$ASCEND_HOME_PATH/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py" "$script_path/CMakePresets.json")
cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

<<<<<<< HEAD
=======
<<<<<<< HEAD
if [ "$cmake_version" \< "3.19.0" ] ; then
    cmake -S . -B "$BUILD_DIR" "$opts"
=======
>>>>>>> lesson1
cmake_major=$(echo "$cmake_version" | cut -d. -f1)
cmake_minor=$(echo "$cmake_version" | cut -d. -f2)
if [ "$cmake_major" -lt 3 ] || { [ "$cmake_major" -eq 3 ] && [ "$cmake_minor" -lt 19 ]; }; then
    cmake -S . -B "$BUILD_DIR" $opts
>>>>>>> lesson3
else
    cmake -S . -B "$BUILD_DIR" --preset=default
fi
cmake --build "$BUILD_DIR" --target binary package -j"$(nproc)"