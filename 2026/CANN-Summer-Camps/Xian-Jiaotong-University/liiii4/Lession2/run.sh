#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " 1. Loading CANN environment"
echo "=========================================="
if [ -n "$ASCEND_TOOLKIT_HOME" ] && [ -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    source "$ASCEND_TOOLKIT_HOME/set_env.sh"
elif [ -n "$ASCEND_HOME_PATH" ] && [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
    source "$ASCEND_HOME_PATH/set_env.sh"
else
    echo "Error: Cannot find set_env.sh. Please check your CANN installation."
    exit 1
fi
echo "CANN environment loaded."

echo "=========================================="
echo " 2. Building custom operator"
echo "=========================================="
cd custom_op
rm -rf build_out
bash build.sh

echo "=========================================="
echo " 3. Installing custom operator"
echo "=========================================="
RUN_FILE=$(ls build_out/custom_opp*.run 2>/dev/null | head -n 1)

if [ -z "$RUN_FILE" ]; then
    echo "Error: .run file not found in build_out. Build may have failed."
    exit 1
fi
echo "Installer: $RUN_FILE"
"$RUN_FILE" --install-path="${HOME}/"
echo "Custom operator installed."

echo "=========================================="
echo " 4. Loading custom operator environment"
echo "=========================================="
CUSTOM_OP_ENV="${HOME}/vendors/customize/bin/set_env.bash"
if [ -f "$CUSTOM_OP_ENV" ]; then
    source "$CUSTOM_OP_ENV"
    echo "Custom operator environment loaded."
else
    echo "Warning: custom operator env script not found at $CUSTOM_OP_ENV"
fi

echo "=========================================="
echo " 5. Building test case"
echo "=========================================="
cd "$SCRIPT_DIR"
g++ -I"$ASCEND_TOOLKIT_HOME/include" \
    -I"${HOME}/vendors/customize/op_api/include" \
    -L"$ASCEND_TOOLKIT_HOME/lib64" \
    -L"${HOME}/vendors/customize/op_api/lib" \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_sub_op
echo "Test case built."

echo "=========================================="
echo " 6. Running test case"
echo "=========================================="
./execute_sub_op
