#!/bin/bash

set -e


# 1. 初始化 CANN 环境变量

if [ -z "$ASCEND_TOOLKIT_HOME" ]; then
    echo "ASCEND_TOOLKIT_HOME is not set"
    exit 1
fi


source ${ASCEND_TOOLKIT_HOME}/set_env.sh

echo "Environment initialization process completed successfully!"



# 2. 检查工程目录

if [ ! -d "custom_op" ]; then
    echo "custom_op directory not found"
    exit 1
fi



# 3. 配置 CANN 路径

sed -i \
'/\"ASCEND_CANN_PACKAGE_PATH\": {/,/}/ s|\"value\": \".*\"|\"value\": \"'"${ASCEND_TOOLKIT_HOME}"'\"|' \
custom_op/CMakePresets.json



# 4. 编译自定义算子

cd custom_op

bash build.sh



PACKAGE=$(ls build_out/custom_opp*.run | head -n 1)


if [ ! -f "${PACKAGE}" ]; then
    echo "custom operator package not found"
    exit 1
fi



${PACKAGE} --install-path=${HOME}


cd ..



# 5. 编译测试程序


g++ \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -I${HOME}/vendors/customize/op_api/include \
    -L${ASCEND_TOOLKIT_HOME}/lib64 \
    -L${HOME}/vendors/customize/op_api/lib \
    test/main.cpp \
    -lcust_opapi \
    -lnnopbase \
    -lacl_rt \
    -o execute_op



# 6. 设置自定义算子环境

source ${HOME}/vendors/customize/bin/set_env.bash



# 7. 性能测试

rm -rf prof

mkdir -p prof


msprof op --output=prof ./execute_op



# 8. 解析性能数据

python3 << 'PY'

import pandas as pd
import glob
import sys


files = glob.glob("prof/*/OpBasicInfo.csv")


if not files:
    print("OpBasicInfo.csv not found")
    sys.exit(1)


df = pd.read_csv(files[0])


pd.set_option(
    'display.max_columns',
    None)

pd.set_option(
    'display.max_rows',
    None)

pd.set_option(
    'display.width',
    None)

pd.set_option(
    'display.max_colwidth',
    None)


print(df)

PY