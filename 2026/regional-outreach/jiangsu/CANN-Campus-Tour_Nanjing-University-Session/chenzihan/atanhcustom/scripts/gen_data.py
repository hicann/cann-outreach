# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# 测试数据生成脚本 - Atanh 算子（float16，4D shape ND Format）
# ============================================================================
#
# 支持预设 4D shape 和自定义 dims
# ============================================================================

import numpy as np
import os
import argparse

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 预设 4D shape
SHAPES = [
    (1, 4, 16, 64),      # 4096 元素
    (2, 8, 32, 128),     # 65536 元素
    (4, 16, 64, 256),    # 1M 元素
]

parser = argparse.ArgumentParser(description="Generate test data for Atanh operator")
parser.add_argument("--shape-idx", type=int, default=0, choices=[0, 1, 2],
                    help="Shape index: 0=[1,4,16,64], 1=[2,8,32,128], 2=[4,16,64,256]")
parser.add_argument("--dims", type=str, default=None,
                    help="Custom 4D shape, e.g. '3 16 32 64'")
args = parser.parse_args()

if args.dims:
    dims = [int(d) for d in args.dims.split()]
    if len(dims) != 4:
        print(f"ERROR: --dims requires exactly 4 values, got {len(dims)}")
        sys.exit(1)
    shape = tuple(dims)
else:
    shape = SHAPES[args.shape_idx]

total_length = 1
for d in shape:
    total_length *= d
dtype = np.float16

print(f"Generating test data: shape={shape}, total={total_length} elements, dtype={dtype}")

# 生成输入数据：atanh 定义域为 |x| < 1
# 使用 tanh 生成值确保在 (-1, 1) 范围内，再 clamp 远离边界避免除零
x = np.tanh(np.random.randn(*shape).astype(np.float64)).astype(dtype)
# clamp 到 [-0.9995, 0.9995] 避免 atanh(±1) = ±inf
x = np.clip(x, -0.9995, 0.9995)

# 插入特定测试值
x_flat = x.flatten()
# 正数、负数、零、接近边界
x_flat[0] = np.float16(0.5)
if len(x_flat) > 1:
    x_flat[1] = np.float16(-0.3)
if len(x_flat) > 2:
    x_flat[2] = np.float16(0.0)
if len(x_flat) > 3:
    # 接近 1 但不等于 1（atanh(0.999) 约为 3.8，float16 可表示）
    x_flat[3] = np.float16(0.99)
if len(x_flat) > 4:
    x_flat[4] = np.float16(-0.99)
if len(x_flat) > 5:
    x_flat[5] = np.float16(0.1)
if len(x_flat) > 6:
    x_flat[6] = np.float16(-0.1)
x = x_flat.reshape(shape)

x.tofile("input/input_x.bin")

# 计算 golden
golden = compute_golden(x)
golden.tofile("output/golden.bin")

print(f"  input/input_x.bin: shape={x.shape}, dtype={x.dtype}")
print(f"  output/golden.bin: shape={golden.shape}, dtype={golden.dtype}")
print(f"  Sample: x[0,0,0,0]={x.flatten()[0]}, golden={golden.flatten()[0]}")
