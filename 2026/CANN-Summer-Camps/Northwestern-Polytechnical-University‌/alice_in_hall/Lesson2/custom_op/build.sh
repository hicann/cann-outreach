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
    fi
else
    export ASCEND_HOME_PATH="$BASE_LIBS_PATH"
fi

echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
script_path="$(cd "$(dirname "$0")" && pwd)"
cd "$script_path"

BUILD_DIR="build_out"
cmake -E rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

read -r -a opts <<< "$(python3 "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py" "${script_path}/CMakePresets.json")"
cmake_version="$(cmake --version | grep "cmake version" | awk '{print $3}')"

version_lt_319=false
if [ "$(printf '%s\n%s\n' "$cmake_version" "3.19.0" | sort -V | head -n 1)" = "$cmake_version" ] && [ "$cmake_version" != "3.19.0" ]; then
    version_lt_319=true
fi

if [ "$version_lt_319" = true ] ; then
    cmake -S . -B "$BUILD_DIR" "${opts[@]}"
else
    cmake -S . -B "$BUILD_DIR" --preset=default
fi
cmake --build "$BUILD_DIR" --target binary package -j"$(nproc)"
