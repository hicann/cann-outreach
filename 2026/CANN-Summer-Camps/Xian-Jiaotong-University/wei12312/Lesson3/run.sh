#!/bin/bash
set -euo pipefail

: "${ASCEND_TOOLKIT_HOME:?Please set ASCEND_TOOLKIT_HOME first.}"
: "${HOME:?HOME is not set.}"

SCRIPT_DIR="$(realpath "$(dirname "$0")")"

# 1. 初始化 CANN 环境变量
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"
echo "🎉 Environment initialization process completed successfully!"

# 2. 编译并部署自定义算子
(
    cd "${SCRIPT_DIR}/custom_op"
    bash build.sh
)

shopt -s nullglob
packages=("${SCRIPT_DIR}"/custom_op/build_out/custom_opp*.run)
shopt -u nullglob
if [[ ${#packages[@]} -ne 1 ]]; then
    echo "Expected exactly one custom operator package, found ${#packages[@]}." >&2
    exit 1
fi
"${packages[0]}" --install-path="${HOME}"

# 3. 编译测试代码
g++ "-I${ASCEND_TOOLKIT_HOME}/include" \
    "-I${HOME}/vendors/customize/op_api/include" \
    "-L${ASCEND_TOOLKIT_HOME}/lib64" \
    "-L${HOME}/vendors/customize/op_api/lib" \
    "${SCRIPT_DIR}/test/main.cpp" \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o "${SCRIPT_DIR}/execute_op"

# 4. 创建性能数据目录并运行 msprof
PROF_DIR="${SCRIPT_DIR}/prof"
cmake -E remove_directory "${PROF_DIR}"
cmake -E make_directory "${PROF_DIR}"
source "${HOME}/vendors/customize/bin/set_env.bash"
msprof op --output="${PROF_DIR}" "${SCRIPT_DIR}/execute_op"

# 5. 使用 Python 解析并展示性能数据
PROF_DIR="${PROF_DIR}" python3 -c "
import os
import pandas as pd
import glob

matches = glob.glob(os.path.join(os.environ['PROF_DIR'], '*', 'OpBasicInfo.csv'))
if len(matches) != 1:
    raise RuntimeError(f'Expected exactly one OpBasicInfo.csv, found {len(matches)}')
csv_file = matches[0]
df = pd.read_csv(csv_file)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(df)
"
