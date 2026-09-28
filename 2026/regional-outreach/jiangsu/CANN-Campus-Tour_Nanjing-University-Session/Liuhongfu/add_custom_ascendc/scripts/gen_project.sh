#!/bin/bash
# =============================================================================
# gen_project.sh
#   基于 add_custom.json 调用 msopgen 生成与本地 CANN 版本严格匹配的官方算子工程
#   骨架，然后把本工程的算子实现源码覆盖进去。
#
#   之所以要走这一步：msopgen 生成的 cmake/ 构建脚本是随 CANN 版本变化的，
#   手写一份 CMakeLists 很难同时兼容 CANN 7.x / 8.x。用本机 msopgen 生成骨架
#   再覆盖源码，是版本兼容性最好的做法。
#
# 用法:
#   SOC_VERSION=Ascend910B2 ./scripts/gen_project.sh
#   或  ./scripts/gen_project.sh Ascend910B4
# =============================================================================
set -e

CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC_DIR="${CUR_DIR}"
GEN_DIR="${CUR_DIR}/build_gen"
OP_NAME="AddCustom"

# Atlas 800T A2 常见取值：Ascend910B1 / Ascend910B2 / Ascend910B3 / Ascend910B4
# 请用 `npu-smi info` 或 `python3 -c "import torch_npu;print(torch_npu.npu.get_device_name(0))"` 确认
SOC_VERSION="${1:-${SOC_VERSION:-Ascend910B2}}"

if ! command -v msopgen >/dev/null 2>&1; then
    echo "[ERROR] 未找到 msopgen 命令。"
    echo "        请先执行: source \${ASCEND_HOME_PATH}/set_env.sh"
    echo "        或:       source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    exit 1
fi

echo "[INFO] SOC_VERSION = ${SOC_VERSION}"
rm -rf "${GEN_DIR}"
mkdir -p "${GEN_DIR}"

# ---------------------------------------------------------------------------
# 1. 生成官方工程骨架
# ---------------------------------------------------------------------------
msopgen gen \
    -i "${SRC_DIR}/add_custom.json" \
    -c "ai_core-${SOC_VERSION}" \
    -lan cpp \
    -out "${GEN_DIR}/${OP_NAME}"

TARGET="${GEN_DIR}/${OP_NAME}"
if [ ! -d "${TARGET}/op_host" ] || [ ! -d "${TARGET}/op_kernel" ]; then
    echo "[ERROR] msopgen 生成的工程结构不符合预期，请检查 ${TARGET}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. 覆盖 op_host / op_kernel 实现
# ---------------------------------------------------------------------------
cp -f "${SRC_DIR}/op_host/add_custom.cpp"      "${TARGET}/op_host/add_custom.cpp"
cp -f "${SRC_DIR}/op_host/add_custom_tiling.h" "${TARGET}/op_host/add_custom_tiling.h"
cp -f "${SRC_DIR}/op_kernel/add_custom.cpp"    "${TARGET}/op_kernel/add_custom.cpp"
echo "[INFO] 已覆盖 op_host / op_kernel 实现"

# ---------------------------------------------------------------------------
# 3. 覆盖 op_api（aclnn 两段式接口）
# ---------------------------------------------------------------------------
if [ -d "${TARGET}/op_api" ]; then
    cp -f "${SRC_DIR}"/op_api/aclnn_add_custom.h   "${TARGET}/op_api/"
    cp -f "${SRC_DIR}"/op_api/aclnn_add_custom.cpp "${TARGET}/op_api/"
    cp -f "${SRC_DIR}"/op_api/add_custom.h         "${TARGET}/op_api/"
    cp -f "${SRC_DIR}"/op_api/add_custom.cpp       "${TARGET}/op_api/"
    echo "[INFO] 已覆盖 op_api (aclnn) 实现"
else
    echo "[WARN] 本机 msopgen 未生成 op_api 目录，aclnn 接口不会被编入算子包。"
    echo "       op_api/ 下的 4 个文件已备好，可在生成 aclnn 算子工程后手工放入。"
fi

echo ""
echo "[OK] 工程已生成: ${TARGET}"
echo "     下一步: ${CUR_DIR}/scripts/build.sh ${SOC_VERSION}"
