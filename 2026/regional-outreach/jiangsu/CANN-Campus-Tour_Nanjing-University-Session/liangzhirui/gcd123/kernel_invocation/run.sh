#!/bin/bash
# gcd123 直接调用（kernel_invocation）一键编译运行脚本
#
# 用法：
#   bash run.sh                      # 默认：ascend910b + npu 模式
#   bash run.sh ascend910b npu       # 指定芯片 + npu 真机模式
#   bash run.sh ascend910b cpu       # 指定芯片 + cpu 仿真模式（无需 NPU 硬件）
#   bash run.sh ascend910 npu        # 其他芯片：ascend910 / ascend310p / ascend910b
#
# 数据生成参数（shape）可在脚本内修改，或直接用 gen_data_and_tiling.py 生成。
clear
set -e

# 指向昇腾软件包安装地址
if [ ! "$ASCEND_HOME_DIR" ]; then
    ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    if [ -f "$ASCEND_HOME_DIR/../set_env.sh" ]; then
        source $ASCEND_HOME_DIR/../set_env.sh
    elif [ -f "$ASCEND_HOME_DIR/bin/setenv.bash" ]; then
        source $ASCEND_HOME_DIR/bin/setenv.bash
    fi
fi

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)
cd $CURRENT_DIR

declare -A VersionMap
VersionMap["ascend910"]="Ascend910A"
VersionMap["ascend310p"]="Ascend310P1"
VersionMap["ascend910b"]="Ascend910B1"

SOC_VERSION=$1
if [ "${SOC_VERSION}x" = "x" ]; then
    SOC_VERSION=ascend910b
fi
if [ -z "${VersionMap[$SOC_VERSION]}" ]; then
    echo "ERROR: unsupported SOC_VERSION: $SOC_VERSION (support: ascend910/ascend310p/ascend910b)"
    exit 1
fi
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ASCEND_HOME_DIR/tools/tikicpulib/lib/${VersionMap[$SOC_VERSION]}:$ASCEND_HOME_DIR/toolkit/tools/simulator/${VersionMap[$SOC_VERSION]}/lib

RUN_MODE=$2
if [ "${RUN_MODE}x" = "x" ]; then
    RUN_MODE=npu
fi
if [ "$RUN_MODE" != "npu" ] && [ "$RUN_MODE" != "cpu" ]; then
    echo "ERROR: RUN_MODE must be npu or cpu"
    exit 1
fi

# 1. 生成输入数据 / 黄金输出 / tiling 数据
#    如需换 shape：python3 gen_data_and_tiling.py --self 1,1,1,4096 --other 1,1,1,1
python3 gen_data_and_tiling.py --self 1,3,4,1 --other 2,1,4,8

# 2. 编译
mkdir -p build
cd build
cmake .. \
    -Dproduct_type=$SOC_VERSION \
    -Dcore_type=AiCore \
    -Dinstall_path=$ASCEND_HOME_DIR
make gcd123_${RUN_MODE} VERBOSE=0
cd -

echo "INFO: compile gcd123_${RUN_MODE} succeed!"

# 3. 执行
./gcd123_${RUN_MODE}
if [ $? -ne 0 ]; then
    echo "ERROR: execute gcd123_${RUN_MODE} failed!"
    exit 1
fi
echo "INFO: execute gcd123_${RUN_MODE} succeed!"
