#!/bin/bash
set -e

# 定位脚本目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Loading CANN environment..."
# 加载 CANN 环境变量
if [ -n "$ASCEND_TOOLKIT_HOME" ] && [ -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    source $ASCEND_TOOLKIT_HOME/set_env.sh
elif [ -n "$ASCEND_HOME_PATH" ] && [ -f "$ASCEND_HOME_PATH/set_env.sh" ]; then
    source $ASCEND_HOME_PATH/set_env.sh
else
    echo "Error: Cannot find set_env.sh. Please check your CANN installation."
    exit 1
fi
echo "CANN environment loaded."

echo "Building custom operator..."
cd custom_op
# 清理构建目录
rm -rf build_out
bash build.sh

RUN_FILE=$(ls build_out/custom_opp*.run 2>/dev/null | head -n 1)

if [ -z "$RUN_FILE" ]; then
    echo "Error: installer not found in build_out." >&2
    exit 1
fi
echo "Installing custom operator: $RUN_FILE"
# 安装算子到用户目录
$RUN_FILE --install-path=${HOME}/

# 加载自定义算子环境变量
if [ -f "${HOME}/vendors/customize/bin/set_env.bash" ]; then
    source ${HOME}/vendors/customize/bin/set_env.bash
else
    echo "Warning: ${HOME}/vendors/customize/bin/set_env.bash not found." >&2
fi

cd "$SCRIPT_DIR"
# 编译测试代码
echo "Building test..."
g++ -I$ASCEND_TOOLKIT_HOME/include \
    -I${HOME}/vendors/customize/op_api/include \
    -L$ASCEND_TOOLKIT_HOME/lib64 \
    -L${HOME}/vendors/customize/op_api/lib \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_sub_op

echo "Running test..."
./execute_sub_op
