#!/bin/bash
# ============================================================
# atanh_custom 算子编译脚本
# ------------------------------------------------------------
# 功能：
#   1. 编译算子动态库（op_host 侧 + op_kernel 侧）
#   2. 编译 Host 侧测试程序（testcase/main.cpp，aclnn 单算子调用）
#
# 用法：
#   bash build.sh [SOC_VERSION] [RUN_MODE]
#   - SOC_VERSION：芯片型号，默认 ascend910b（可传 ascend310p/ascend910/ascend910c 等）
#   - RUN_MODE   ：cpu（CPU 侧调测）/ npu（NPU 侧运行），默认 npu
#
# 前置条件：
#   - 已安装 CANN 工具包，并正确设置 ASCEND_CANN_PACKAGE_PATH
#   - 算子已按本工程结构组织（op_host / op_kernel）
# ============================================================
set -e

# ---------- 参数解析 ----------
SOC_VERSION=${1:-ascend910b}
RUN_MODE=${2:-npu}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR=${SCRIPT_DIR}/build

# ---------- 1. 设置 CANN 环境 ----------
if [ -z "${ASCEND_CANN_PACKAGE_PATH}" ]; then
    ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/ascend-toolkit/latest
    export ASCEND_CANN_PACKAGE_PATH
fi

# 不同 CANN 版本的 set_env 脚本路径略有差异，依次尝试
if [ -f "${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash" ]; then
    source "${ASCEND_CANN_PACKAGE_PATH}/bin/setenv.bash"
elif [ -f "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh" ]; then
    source "${ASCEND_CANN_PACKAGE_PATH}/set_env.sh"
elif [ -n "${ASCEND_HOME_PATH}" ] && [ -f "${ASCEND_HOME_PATH}/ascend-toolkit/set_env.sh" ]; then
    source "${ASCEND_HOME_PATH}/ascend-toolkit/set_env.sh"
else
    echo "[WARN] 未找到 set_env 脚本，请确认 ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH} 是否正确"
fi

echo "============================================"
echo ">>> 开始编译算子库 atanh_custom"
echo "    SOC_VERSION = ${SOC_VERSION}"
echo "    RUN_MODE    = ${RUN_MODE}"
echo "============================================"

# ---------- 2. cmake 编译算子动态库 ----------
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DSOC_VERSION=${SOC_VERSION} \
    -DRUN_MODE=${RUN_MODE} \
    -DCMAKE_BUILD_TYPE=Release

make -j"$(nproc)"

echo ""
echo ">>> 算子库编译完成，产物位于 ${BUILD_DIR}/"
ls -lh "${BUILD_DIR}"/*.so 2>/dev/null || \
    echo "[WARN] 未在 build 目录找到 .so，请检查 CMakeLists 配置"

# ---------- 3. 编译 Host 侧测试程序（aclnn 单算子调用） ----------
# 说明：
#   - 算子库编译部署后，会自动生成 aclnn 单算子接口头文件 aclnn_atanh_custom.h
#   - 若未自动部署，请先执行 make install 或按官方流程把算子包安装到 opp 目录
echo ""
echo ">>> 编译 Host 侧测试程序 testcase/main.cpp"

# aclnn 接口头文件与库目录（算子部署后生成）
if [ -z "${ASCEND_OPP_PATH}" ]; then
    ASCEND_OPP_PATH=${ASCEND_CANN_PACKAGE_PATH}/opp
fi
ACLNN_INC=${ASCEND_OPP_PATH}/op_api/include
ACLNN_LIB=${ASCEND_OPP_PATH}/op_api/lib

# CANN 头文件与库目录（不同版本路径略有差异，请按实际调整）
CANN_INC=${ASCEND_CANN_PACKAGE_PATH}/include
CANN_LIB=${ASCEND_CANN_PACKAGE_PATH}/lib64

g++ -std=c++17 "${SCRIPT_DIR}/testcase/main.cpp" -o "${BUILD_DIR}/main" \
    -I"${CANN_INC}" \
    -I"${CANN_INC}/acl" \
    -I"${ACLNN_INC}" \
    -L"${CANN_LIB}" \
    -L"${ACLNN_LIB}" \
    -lascendcl -lcust_opapi \
    -Wl,-rpath,"${CANN_LIB}:${ACLNN_LIB}"

echo ""
echo ">>> 测试程序编译完成：${BUILD_DIR}/main"
echo "============================================"
echo ">>> 全部编译完成。下一步执行：bash run.sh"
echo "============================================"
