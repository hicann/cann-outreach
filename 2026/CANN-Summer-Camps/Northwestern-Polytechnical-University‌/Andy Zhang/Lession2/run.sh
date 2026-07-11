#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

echo "[1/4] Building custom operator"
cd "$SCRIPT_DIR/custom_op"
bash build.sh

echo "[2/4] Installing custom operator"
RUN_FILE="$(find build_out -maxdepth 1 -name 'custom_opp*.run' -print -quit)"
if [[ -z "$RUN_FILE" ]]; then
    echo "Error: custom operator installer was not generated." >&2
    exit 1
fi
"$RUN_FILE" --install-path="$HOME"

CUSTOM_ENV="$HOME/vendors/customize/bin/set_env.bash"
if [[ ! -f "$CUSTOM_ENV" ]]; then
    echo "Error: custom operator environment was not installed." >&2
    exit 1
fi
source "$CUSTOM_ENV"

echo "[3/4] Building test"
cd "$SCRIPT_DIR"
g++ -std=c++11 \
    -I"$ASCEND_TOOLKIT_HOME/include" \
    -I"$HOME/vendors/customize/op_api/include" \
    -L"$ASCEND_TOOLKIT_HOME/lib64" \
    -L"$HOME/vendors/customize/op_api/lib" \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_sub_op

echo "[4/4] Running test"
./execute_sub_op
