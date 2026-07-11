#!/bin/bash
# ============================================================================
# SubCustomTemplate 算子 — 一键编译/部署/测试/Profiling
#
# 用法:
#   bash run.sh                 # 完整流程
#   bash run.sh --skip-profile  # 跳过 profiling, 仅编译+测试
# ============================================================================
set -e

SKIP_PROFILE=0
for arg in "$@"; do
    case "$arg" in
        --skip-profile) SKIP_PROFILE=1 ;;
    esac
done

# 1. 初始化 CANN 环境
source $ASCEND_TOOLKIT_HOME/set_env.sh
echo "==> Environment initialization completed!"

# 2. 配置 CANN 路径
sed -i '/"ASCEND_CANN_PACKAGE_PATH": {/,/}/ s|\s*"value": ".*"| "value": "'"$ASCEND_TOOLKIT_HOME"'"|' custom_op/CMakePresets.json

# 3. 编译并部署自定义算子
cd custom_op
bash build.sh
./build_out/custom_opp*.run --install-path=${HOME}
cd ..

# 4. 编译测试代码
g++ -I$ASCEND_TOOLKIT_HOME/include \
    -I${HOME}/vendors/customize/op_api/include \
    -L$ASCEND_TOOLKIT_HOME/lib64 \
    -L${HOME}/vendors/customize/op_api/lib \
    test/main.cpp \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o execute_op

# 5. Profiling
if [ "$SKIP_PROFILE" -eq 0 ]; then
    rm -rf prof
    mkdir -p prof
    source ${HOME}/vendors/customize/bin/set_env.bash
    msprof op --output=prof ./execute_op

    # 6. 解析 profiling 结果
    python3 -c "
import pandas as pd
import glob

csv_file = glob.glob('prof/*/OpBasicInfo.csv')[0]
df = pd.read_csv(csv_file)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df)
"
else
    source ${HOME}/vendors/customize/bin/set_env.bash
    ./execute_op
fi
