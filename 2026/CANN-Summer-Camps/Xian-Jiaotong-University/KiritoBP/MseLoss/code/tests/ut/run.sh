#!/bin/bash
# mse_loss 算子 UT 测试执行脚本
# 格式化输出：Google Test 风格 [ OK ] / [ FAILED ]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "========================================"
echo "  mse_loss 算子 Kernel UT 测试"
echo "========================================"
echo ""

# --- Env ---
if [ -z "$ASCEND_HOME_PATH" ]; then
    export ASCEND_HOME_PATH=/home/developer/Ascend/cann
fi
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

# --- Build ---
echo "[----------] Building test binaries"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. > /dev/null 2>&1
make -j$(nproc) > /dev/null 2>&1
echo "[----------] Build complete"
echo ""

# --- Data ---
echo "[----------] Generating test data"
DATA_DIR="${SCRIPT_DIR}/op_kernel/mse_loss_data"
cd "${DATA_DIR}"
python3 gen_data.py 2>&1 | sed 's/^/  /'
echo ""

# --- Run ---
START_TIME=$(date +%s%N)

echo "[==========] Running 1 test from 1 test suite."
echo "[----------] Global test environment set-up."
echo ""

echo -n "[ RUN      ] MseLossKernelTest.test_kernel_run"

cd "${BUILD_DIR}/op_kernel"
RUN_OUT=$(./mse_loss_op_kernel_ut 2>&1)
RUN_EXIT=$?

# Accuracy check
cd "${DATA_DIR}"
ACC_OUT=$(python3 compare_data.py 2>&1)
ACC_EXIT=$?

END_TIME=$(date +%s%N)
ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

echo ""

if [ $RUN_EXIT -eq 0 ] && [ $ACC_EXIT -eq 0 ]; then
    echo "MseLossKernelTest SetUp"
    echo "MseLossKernelTest TearDown"
    echo "[       OK ] MseLossKernelTest.test_kernel_run (${ELAPSED_MS} ms)"
    echo "[----------] 1 test from MseLossKernelTest (${ELAPSED_MS} ms total)"
    echo ""
    echo "[----------] Global test environment tear-down"
    echo "[==========] 1 test from 1 test suite ran. (${ELAPSED_MS} ms total)"
    echo "[  PASSED  ] 1 test."
    exit 0
else
    echo "[  FAILED  ] MseLossKernelTest.test_kernel_run (${ELAPSED_MS} ms)"
    echo "[----------] 1 test from MseLossKernelTest (${ELAPSED_MS} ms total)"
    echo ""
    echo "[----------] Global test environment tear-down"
    echo "[==========] 1 test from 1 test suite ran. (${ELAPSED_MS} ms total)"
    echo "[  FAILED  ] 1 test, listed below:"
    echo "[  FAILED  ] MseLossKernelTest.test_kernel_run"
    echo ""
    echo " 1 FAILED TEST"
    if [ $RUN_EXIT -ne 0 ]; then
        echo "Kernel run output:"
        echo "$RUN_OUT" | sed 's/^/  /'
    fi
    if [ $ACC_EXIT -ne 0 ]; then
        echo "Accuracy check output:"
        echo "$ACC_OUT" | sed 's/^/  /'
    fi
    exit 1
fi
