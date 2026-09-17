#!/bin/bash
# ---------------------------------------------------------------------------
# add_custom 算子一键构建 + 运行 + 精度校验脚本（在 Ascend 环境上执行）
#
# 用法：bash run.sh [N2] [N1] [blockDim]
#   N2       shape 第 0 维，默认 8
#   N1       shape 第 1 维，默认 2048
#   blockDim 使用的 AI Core 个数，默认 8
# ---------------------------------------------------------------------------
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

N2="${1:-8}"
N1="${2:-2048}"
BLOCK_DIM="${3:-8}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# 1. 加载 CANN 环境变量（若当前 shell 尚未加载）
if [ -z "${ASCEND_HOME_PATH}" ]; then
    for env_script in \
        "/usr/local/Ascend/ascend-toolkit/set_env.sh" \
        "/usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash"; do
        if [ -f "${env_script}" ]; then
            echo "[INFO] source ${env_script}"
            # shellcheck disable=SC1090
            source "${env_script}"
            break
        fi
    done
fi

if [ -z "${ASCEND_HOME_PATH}" ]; then
    echo "[ERROR] ASCEND_HOME_PATH 未设置，请先 source CANN 的 set_env.sh"
    exit 1
fi
echo "[INFO] ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}"

# 2. 生成输入数据与真值
mkdir -p input output
${PYTHON_BIN} scripts/gen_data.py "${N2}" "${N1}"

# 3. 编译
mkdir -p build
cd build
cmake ..
make -j"$(nproc 2>/dev/null || echo 4)"
cd "${SCRIPT_DIR}"

# 4. 运行算子（Host 侧会自动与 golden_z.bin 做精度比对）
./build/add_custom "${N2}" "${N1}" "${BLOCK_DIM}"

# 5. 独立精度校验（不依赖算子进程内的比对结果）
${PYTHON_BIN} scripts/verify_result.py "${N2}" "${N1}"

echo "[INFO] all done"
