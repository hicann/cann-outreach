#!/bin/bash
# =============================================================================
# 路线 A：把 InplaceRsqrt 编译成可安装的自定义算子包（custom_opp_<soc>.run）
#
#   用法: bash build.sh [soc_version]
#         soc_version 例如 ascend910b / ascend910_93 / ascend310p，默认 ascend910b
#
#   产物: build/skeleton/<工程目录>/build_out/custom_opp_*.run
#   安装: ./build_out/custom_opp_*.run --install
# =============================================================================
set -e

SOC_VERSION=${1:-ascend910b}
ROOT=$(cd "$(dirname "$0")" && pwd)
OUT=${ROOT}/build/skeleton

# 1) 准备 CANN 环境（msopgen / cmake 工具链都在 CANN 里）
if [ -z "${ASCEND_HOME_PATH}" ]; then
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    else
        echo "ERROR: 请先 source <CANN>/set_env.sh 设置 ASCEND_HOME_PATH" >&2
        exit 1
    fi
fi
echo "ASCEND_HOME_PATH = ${ASCEND_HOME_PATH}, soc = ${SOC_VERSION}"

# 2) 生成工程骨架：cmake / OpDef / aclnn 适配代码都按本机 CANN 版本自动生成
mkdir -p "${OUT}"
msopgen gen -i "${ROOT}/InplaceRsqrt.json" -c ai_core-${SOC_VERSION} -lan cpp -out "${OUT}"

# 定位生成的工程目录（含 build.sh 的那一层）
PROJ=$(find "${OUT}" -maxdepth 2 -name build.sh -printf '%h\n' 2>/dev/null | head -n 1)
if [ -z "${PROJ}" ]; then
    PROJ="${OUT}/InplaceRsqrt"
fi
echo "PROJ = ${PROJ}"

# 3) 用本仓库的实现覆盖骨架中的实现文件
#    （骨架里与算子同名的实现文件会被直接覆盖，不会残留两份实现）
cp -f "${ROOT}/op_kernel/inplace_rsqrt.h"      "${PROJ}/op_kernel/"
cp -f "${ROOT}/op_kernel/inplace_rsqrt.cpp"    "${PROJ}/op_kernel/"
cp -f "${ROOT}/op_host/inplace_rsqrt_tiling.h" "${PROJ}/op_host/"
cp -f "${ROOT}/op_host/inplace_rsqrt.cpp"      "${PROJ}/op_host/"

echo "--- op_kernel ---"; ls -l "${PROJ}/op_kernel"
echo "--- op_host ---";   ls -l "${PROJ}/op_host"

# 4) 编译（骨架里的 build.sh 由 CANN 提供，参数无需改动）
cd "${PROJ}"
bash build.sh

echo "----------------------------------------------------------------"
echo "编译完成，产物："
ls -l ./build_out/*.run
echo "安装：./build_out/custom_opp_*.run --install"