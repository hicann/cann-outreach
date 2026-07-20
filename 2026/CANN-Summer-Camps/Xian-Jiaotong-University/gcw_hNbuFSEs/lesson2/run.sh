#!/bin/bash
set -e

# Locate this script so the command works from any directory.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " 1. Loading CANN Environment"
echo "=========================================="
# Load CANN environment variables.
if [ -n "$ASCEND_TOOLKIT_HOME" ] && [ -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    source $ASCEND_TOOLKIT_HOME/set_env.sh
elif [ -n "$ASCEND_HOME_PATH" ] && [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
    source $ASCEND_HOME_PATH/set_env.sh
else
    echo "Error: Cannot find set_env.sh. Please check your CANN installation."
    exit 1
fi
echo "[OK] CANN environment loaded successfully."

echo "=========================================="
echo " 2. Building Custom Operator"
echo "=========================================="
cd custom_op
# Remove the old build directory for a clean build.
rm -rf build_out
echo ">>> Running build.sh..."
bash build.sh

echo "=========================================="
echo " 3. Installing Custom Operator"
echo "=========================================="
RUN_FILE=$(ls build_out/custom_opp*.run 2>/dev/null | head -n 1)

if [ -z "$RUN_FILE" ]; then
    echo "[ERROR] .run file not found in build_out. Build might have failed."
    exit 1
fi
echo ">>> Found installer: $RUN_FILE"
# Install the operator into the user directory.
$RUN_FILE --install-path=${HOME}/
echo "[OK] Operator installed successfully."

echo "=========================================="
echo " 4. Loading Custom Operator Environment"
echo "=========================================="
# Source the custom operator environment so runtime can find the libraries.
if [ -f "${HOME}/vendors/customize/bin/set_env.bash" ]; then
    source ${HOME}/vendors/customize/bin/set_env.bash
    echo "[OK] Custom operator environment loaded."
else
    echo "[WARN] Custom operator env script not found at ${HOME}/vendors/customize/bin/set_env.bash"
fi

echo "=========================================="
echo " 5. Building Test Case"
echo "=========================================="
cd "$SCRIPT_DIR"
# Build the ACL test case.
echo ">>> Compiling test/main.cpp..."
g++ -I$ASCEND_TOOLKIT_HOME/include \
    -I${HOME}/vendors/customize/op_api/include \
    -L$ASCEND_TOOLKIT_HOME/lib64 \
    -L${HOME}/vendors/customize/op_api/lib \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_sub_op
echo "[OK] Test case built successfully."

echo "=========================================="
echo " 6. Running Test Case"
echo "=========================================="
echo ">>> Executing..."
./execute_sub_op
