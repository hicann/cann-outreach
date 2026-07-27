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

if [ -n "${BASE_LIBS_PATH:-}" ]; then
    export ASCEND_HOME_PATH="${BASE_LIBS_PATH}"
elif [ -n "${ASCEND_HOME_PATH:-}" ]; then
    export ASCEND_HOME_PATH
elif [ -n "${ASCEND_AICPU_PATH:-}" ]; then
    export ASCEND_HOME_PATH="${ASCEND_AICPU_PATH}"
else
    echo "please set BASE_LIBS_PATH, ASCEND_HOME_PATH or ASCEND_AICPU_PATH."
    exit 1
fi
echo "using ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}"
script_path="$(realpath "$(dirname "$0")")"
cd "${script_path}"

BUILD_DIR="${script_path}/build_out"
mkdir -p "${BUILD_DIR}"
rm -rf -- "${BUILD_DIR:?}/"*
opts_output="$(python3 "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake/fwk_modules/util/preset_parse.py" \
    "${script_path}/CMakePresets.json")"
opts=()
if [ -n "${opts_output}" ]; then
    read -r -a opts <<< "${opts_output}"
fi
cmake_version="$(cmake --version | awk 'NR == 1 {print $3}')"

if [ "${cmake_version}" \< "3.19.0" ] ; then
    cmake -S "${script_path}" -B "${BUILD_DIR}" "${opts[@]}"
else
    cmake -S "${script_path}" -B "${BUILD_DIR}" --preset=default
fi
cmake --build "${BUILD_DIR}" --target binary package -j"$(nproc)"
