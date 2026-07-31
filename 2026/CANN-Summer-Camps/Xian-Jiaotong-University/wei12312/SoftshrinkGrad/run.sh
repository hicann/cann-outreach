#!/bin/bash
set -euo pipefail

: "${ASCEND_TOOLKIT_HOME:?Please set ASCEND_TOOLKIT_HOME first.}"
: "${HOME:?HOME is not set.}"

SCRIPT_DIR="$(realpath "$(dirname "$0")")"

python3 "${SCRIPT_DIR}/test/test_reference.py"

(
    cd "${SCRIPT_DIR}/custom_op"
    bash build.sh
)

shopt -s nullglob
packages=("${SCRIPT_DIR}"/custom_op/build_out/custom_opp*.run)
shopt -u nullglob
if [[ ${#packages[@]} -ne 1 ]]; then
    echo "Expected one custom operator package, found ${#packages[@]}." >&2
    exit 1
fi

"${packages[0]}" --install-path="${HOME}"
export CUSTOM_OPP_PATH="${HOME}/vendors/customize"

# CANN-generated vendor environment scripts may read optional variables before
# assigning them. Initialize the known variable and temporarily relax nounset
# while sourcing the external script.
export ASCEND_CUSTOM_OPP_PATH="${ASCEND_CUSTOM_OPP_PATH:-}"
set +u
source "${CUSTOM_OPP_PATH}/bin/set_env.bash"
set -u

bash "${SCRIPT_DIR}/custom_op/test/run.sh"
