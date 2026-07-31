#!/bin/bash
# truncate_mod 算子 UT 测试执行脚本（跳过 op_host UT，聚焦 kernel + 精度比对）

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
DATA_DIR="${SCRIPT_DIR}/op_kernel/truncate_mod_data"

echo "========================================"
echo "truncate_mod 算子 UT 测试"
echo "========================================"

if [ -z "$ASCEND_HOME_PATH" ]; then
    export ASCEND_HOME_PATH=/home/developer/Ascend/cann
fi
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH}

PYTHON=/usr/local/python3.12.13/bin/python3

# 清理旧构建
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake .. -DSOC_VERSION=Ascend910B1
make -j$(nproc)

echo "========================================"
echo "生成测试数据"
echo "========================================"
cd "${DATA_DIR}"
$PYTHON gen_data.py

echo "========================================"
echo "执行 Kernel UT"
echo "========================================"
cd "${BUILD_DIR}/op_kernel"
./truncate_mod_op_kernel_ut

# 拷贝输出 bin 到 DATA_DIR
cp ./*_output_*.bin "${DATA_DIR}/" 2>/dev/null || true

echo "========================================"
echo "精度比对"
echo "========================================"
cd "${DATA_DIR}"
$PYTHON compare_data.py

echo "========================================"
echo "测试完成"
echo "========================================"
