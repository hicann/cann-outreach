#!/bin/bash
set -euo pipefail
# 1. 初始化 CANN 环境变量
if [ -z "${ASCEND_TOOLKIT_HOME:-}" ]; then
    echo "ASCEND_TOOLKIT_HOME is not set."
    exit 1
fi

source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
export ASCEND_CANN_PACKAGE_PATH="${ASCEND_TOOLKIT_HOME}"
echo "🎉 Environment initialization process completed successfully!"

# 2. 将算子工程中的 CANN 路径配置为实际路径

# 3. 编译并部署自定义算子
cd custom_op || { echo "Failed to cd to custom_op"; exit 1; }
bash build.sh
./build_out/custom_opp*.run --install-path="${HOME}"
cd ..

# 4. 编译测试代码
g++ -I"${ASCEND_TOOLKIT_HOME}/include" \
    -I"${HOME}"/vendors/customize/op_api/include \
    -L"${ASCEND_TOOLKIT_HOME}/lib64" \
    -L"${HOME}"/vendors/customize/op_api/lib \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_op

# 5. 创建性能数据目录并运行 msprof
rm -rf prof
mkdir -p prof
source "${HOME}"/vendors/customize/bin/set_env.bash
msprof op --output=prof ./execute_op

# 6. 使用 Python 解析并展示性能数据
python3 -c "
import pandas as pd
import glob

csv_file = (glob.glob('prof/*/OpBasicInfo.csv') or [None])[0]
df = pd.read_csv(csv_file)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df)
"