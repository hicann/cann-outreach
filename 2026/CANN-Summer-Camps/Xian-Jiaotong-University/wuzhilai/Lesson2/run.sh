#!/bin/bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "1. Loading CANN Environment"
echo "=========================================="
if [ -n "${ASCEND_TOOLKIT_HOME:-}" ] && [ -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    TOOLKIT_HOME="$ASCEND_TOOLKIT_HOME"
elif [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
    TOOLKIT_HOME="$ASCEND_HOME_PATH"
else
    echo "Error: Cannot find set_env.sh. Please check your CANN installation."
    exit 1
fi
source "$TOOLKIT_HOME/set_env.sh"
TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$TOOLKIT_HOME}"
echo "CANN environment loaded successfully."

echo "=========================================="
echo "2. Building Custom Operator"
echo "=========================================="
cd "$SCRIPT_DIR/custom_op"
rm -rf build_out
echo ">>> Running build.sh..."
bash build.sh

echo "=========================================="
echo "3. Installing Custom Operator"
echo "=========================================="
RUN_FILE="$(find build_out -maxdepth 1 -type f -name 'custom_opp*.run' -print -quit)"
if [ -z "$RUN_FILE" ]; then
    echo "Error: .run file not found in build_out. Build might have failed."
    exit 1
fi
echo ">>> Found installer: $RUN_FILE"
"$RUN_FILE" --install-path="${HOME}/"
echo "Operator installed successfully."

echo "=========================================="
echo "4. Loading Custom Operator Environment"
echo "=========================================="
CUSTOM_ENV="${HOME}/vendors/customize/bin/set_env.bash"
if [ -f "$CUSTOM_ENV" ]; then
    source "$CUSTOM_ENV"
    echo "Custom operator environment loaded."
else
    echo "Warning: custom operator environment script was not found at $CUSTOM_ENV"
fi

echo "=========================================="
echo "5. Building Test Case"
echo "=========================================="
cd "$SCRIPT_DIR"
echo ">>> Compiling test/main.cpp..."
g++ -I"$TOOLKIT_HOME/include" \
    -I"${HOME}/vendors/customize/op_api/include" \
    -L"$TOOLKIT_HOME/lib64" \
    -L"${HOME}/vendors/customize/op_api/lib" \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_sub_op
echo "Test case built successfully."

echo "=========================================="
echo "6. Running Test Case"
echo "=========================================="
echo ">>> Executing..."
./execute_sub_op
