#!/bin/bash
set -e

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " 1. Loading CANN Environment"
echo "=========================================="
if [ -n "${ASCEND_TOOLKIT_HOME:-}" ] && [ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]; then
    source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
elif [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/set_env.sh"
else
    if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    elif [ -f "/usr/local/Ascend/cann-8.5.1/set_env.sh" ]; then
        source /usr/local/Ascend/cann-8.5.1/set_env.sh
    else
        echo "Error: Cannot find set_env.sh."
        exit 1
    fi
fi
echo "CANN Environment loaded."

echo "=========================================="
echo " 2. Generating Operator Project (msopgen)"
echo "=========================================="
if [ -d "custom_op" ] && [ -f "custom_op/build.sh" ]; then
    echo ">>> custom_op exists, skip msopgen."
else
    echo ">>> Generating operator project..."
    SOC_VERSION=$(python3 -c "import acl; print(acl.get_soc_name())" 2>/dev/null || echo "Ascend910B")
    if echo "$SOC_VERSION" | grep -qi "Ascend910"; then
        SOC_PARAM="ai_core-ascend910a"
    else
        SOC_PARAM="ai_core-ascend910b1"
    fi
    echo ">>> SoC: $SOC_VERSION, param: $SOC_PARAM"
    msopgen gen -i div_custom_template.json -c $SOC_PARAM -lan cpp -out ./custom_op
    echo ">>> Generated. Please implement kernel in custom_op/op_kernel, then re-run."
    exit 0
fi

echo "=========================================="
echo " 3. Compiling Operator"
echo "=========================================="
cd custom_op
bash build.sh 2>&1
cd ..

echo "=========================================="
echo " 4. Installing Operator"
echo "=========================================="
RUN_FILE=$(ls custom_op/build_out/custom_*.run 2>/dev/null | head -1)
if [ -z "$RUN_FILE" ]; then
    echo "Error: No .run file found."
    exit 1
fi
$RUN_FILE --install-path=${HOME}/

VENDOR_PATH=${HOME}/vendors/customize
TILING_LIB=$VENDOR_PATH/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64
PROTO_LIB=$VENDOR_PATH/op_proto/lib/linux/aarch64
if [ -d "$TILING_LIB" ]; then
    export LD_LIBRARY_PATH=$TILING_LIB:$LD_LIBRARY_PATH
fi
if [ -d "$PROTO_LIB" ]; then
    export LD_LIBRARY_PATH=$PROTO_LIB:$LD_LIBRARY_PATH
fi

echo "=========================================="
echo " 5. Building Test"
echo "=========================================="
g++ -I"${ASCEND_TOOLKIT_HOME}/include" \
    -I"${VENDOR_PATH}/op_api/include" \
    -L"${ASCEND_TOOLKIT_HOME}/lib64" \
    -L"${VENDOR_PATH}/op_api/lib" \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_div_op

echo "=========================================="
echo " 6. Running Test"
echo "=========================================="
./execute_div_op
