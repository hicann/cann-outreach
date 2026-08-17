#!/usr/bin/env bash
set -euo pipefail

# Load CANN when the caller has not already initialized the environment.
if [[ -n "${ASCEND_TOOLKIT_HOME:-}" && -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]]; then
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
elif [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
    source "${ASCEND_HOME_PATH}/set_env.sh"
elif [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    source "${ASCEND_HOME_PATH}/bin/setenv.bash"
else
    echo "[ERROR] CANN environment script was not found." >&2
    exit 1
fi

cmake -S . -B build
cmake --build build --parallel
./build/div_test
