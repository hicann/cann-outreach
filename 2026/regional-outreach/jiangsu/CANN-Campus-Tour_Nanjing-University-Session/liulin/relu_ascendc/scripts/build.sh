#!/bin/bash
# Relu 算子（AscendC, float16, ND, 4维）构建脚本
# 依赖：CANN Toolkit（含 AscendC 编译器），在装有 CANN 的 Linux 环境执行
#   bash scripts/build.sh          # 仅编译
#   bash scripts/build.sh run      # 编译并运行 CPU 模拟验证
set -e
cd "$(dirname "$0")/.."

# 1) 编译 kernel（NPU 侧）
mkdir -p build
ascendc_kernel_compile \
    --platform=ascend910b --save-temp-dir=build/tmp \
    kernel/relu_custom.cpp -o build/relu_custom.o
echo "[OK] kernel compiled: build/relu_custom.o"

# 2) 编译 host 侧验证程序（CPU 模拟模式：-mcpu=armv8.2-a+fp16 或使用框架桩）
if [ "$1" = "run" ]; then
    g++ -std=c++14 -O1 -march=armv8.2-a+fp16 \
        host/main.cpp build/relu_custom.o \
        -o build/relu_test -I host
    echo "[OK] host built, running..."
    ./build/relu_test
fi
