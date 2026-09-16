#!/bin/bash
set -e

# 获取当前脚本所在目录，确保在任何路径下执行都能找到正确文件
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " 1. Loading CANN Environment"
echo "=========================================="
# 加载 CANN 环境变量
if [ -n "${ASCEND_TOOLKIT_HOME:-}" ] && [ -f "${ASCEND_TOOLKIT_HOME}/set_env.sh" ]; then
    CANN_INSTALL_PATH="${ASCEND_TOOLKIT_HOME}"
elif [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
    CANN_INSTALL_PATH="${ASCEND_HOME_PATH}"
else
    echo "Error: Cannot find set_env.sh. Please check your CANN installation."
    exit 1
fi
source "${CANN_INSTALL_PATH}/set_env.sh"
export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-${CANN_INSTALL_PATH}}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${CANN_INSTALL_PATH}}"
echo "CANN Environment loaded successfully."

echo "=========================================="
echo " 2. Checking Operator Project"
echo "=========================================="
if [ ! -d "custom_op" ] || [ ! -f "custom_op/build.sh" ]; then
    echo "Error: custom_op project is missing."
    exit 1
fi
echo ">>> custom_op project found."

echo "=========================================="
echo " 3. Building Custom Operator"
echo "=========================================="
cd custom_op
# 清理旧的构建目录以确保干净编译
rm -rf build_out
echo ">>> Running build.sh..."
bash build.sh

echo "=========================================="
echo " 4. Installing Custom Operator"
echo "=========================================="
RUN_FILE=$(ls build_out/custom_opp*.run 2>/dev/null | head -n 1)

if [ -z "$RUN_FILE" ]; then
    echo "Error: .run file not found in build_out. Build might have failed."
    exit 1
fi
echo ">>> Found installer: $RUN_FILE"
# 安装算子到用户目录
"$RUN_FILE" --install-path="${HOME}/"
echo "Operator installed successfully."

echo "=========================================="
echo " 5. Loading Custom Operator Environment"
echo "=========================================="
# 【关键】必须 source 自定义算子的环境变量，否则运行时找不到算子库
if [ -f "${HOME}/vendors/customize/bin/set_env.bash" ]; then
    source ${HOME}/vendors/customize/bin/set_env.bash
    echo "Custom operator environment loaded."
else
    echo "Warning: Custom operator env script not found at ${HOME}/vendors/customize/bin/set_env.bash"
fi

echo "=========================================="
echo " 6. Building Test Case"
echo "=========================================="
cd "$SCRIPT_DIR"
# 编译测试代码
echo ">>> Compiling test/main.cpp..."
g++ -I"${ASCEND_TOOLKIT_HOME}/include" \
    -I"${HOME}/vendors/customize/op_api/include" \
    -L"${ASCEND_TOOLKIT_HOME}/lib64" \
    -L"${HOME}/vendors/customize/op_api/lib" \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_div_op
echo "Test case built successfully."

echo "=========================================="
echo " 7. Running Test Case"
echo "=========================================="
echo ">>> Executing..."
./execute_div_op
