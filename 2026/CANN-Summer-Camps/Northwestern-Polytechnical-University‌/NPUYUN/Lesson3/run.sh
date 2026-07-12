#!/bin/bash
set -e

# 1. 初始化 CANN 环境变量
if [ -z "${ASCEND_TOOLKIT_HOME:-}" ] && [ -n "${ASCEND_HOME_PATH:-}" ]; then
    export ASCEND_TOOLKIT_HOME="$ASCEND_HOME_PATH"
fi
if [ -z "${ASCEND_TOOLKIT_HOME:-}" ] || [ ! -f "$ASCEND_TOOLKIT_HOME/set_env.sh" ]; then
    echo "Error: ASCEND_TOOLKIT_HOME is not set or set_env.sh is missing."
    exit 1
fi
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
echo "🎉 Environment initialization process completed successfully!"

# 2. 编译并部署自定义算子
cd custom_op
bash build.sh
./build_out/custom_opp*.run --install-path="${HOME}"
cd ..

# 3. 编译测试代码
g++ -I"$ASCEND_TOOLKIT_HOME/include" \
    -I"${HOME}/vendors/customize/op_api/include" \
    -L"$ASCEND_TOOLKIT_HOME/lib64" \
    -L"${HOME}/vendors/customize/op_api/lib" \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_op

# 4. 创建性能数据目录并运行 msprof
rm -rf prof
mkdir -p prof
source "${HOME}/vendors/customize/bin/set_env.bash"
msprof op --output=prof ./execute_op

# 5. 使用 Python 解析并展示性能数据
python3 -c "
import glob
import csv

matches = glob.glob('prof/*/OpBasicInfo.csv')
if not matches:
    raise SystemExit('OpBasicInfo.csv not found under prof')
csv_file = matches[0]
with open(csv_file, newline='') as f:
    for row in csv.reader(f):
        print('\t'.join(row))
"
