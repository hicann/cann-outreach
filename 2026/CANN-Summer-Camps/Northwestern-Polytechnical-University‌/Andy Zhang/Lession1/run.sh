#!/usr/bin/env bash
set -eo pipefail

if [[ -z "${ASCEND_TOOLKIT_HOME:-}" ]]; then
    for candidate in "$HOME/Ascend/cann-8.5.0" \
        "/usr/local/Ascend/ascend-toolkit/latest" \
        "/usr/local/Ascend/ascend-toolkit"; do
        if [[ -f "$candidate/set_env.sh" ]]; then
            ASCEND_TOOLKIT_HOME="$candidate"
            break
        fi
    done
fi

: "${ASCEND_TOOLKIT_HOME:?Unable to locate the Ascend toolkit}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-$ASCEND_TOOLKIT_HOME}"
source "$ASCEND_TOOLKIT_HOME/set_env.sh"

mkdir -p build
export ASC_DIR="$ASCEND_HOME_PATH/aarch64-linux/tikcpp/ascendc_kernel_cmake/"
cmake -S . -B build
cmake --build build -j
./build/mul_test
