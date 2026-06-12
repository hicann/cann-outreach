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
# 测试数据生成脚本 - atanh 算子
# ============================================================================
#
# 生成 4D shape 的 float16 测试数据。
# atanh(x) 定义域: x ∈ (-1, 1)
# 使用 numpy 生成范围内的随机数和精确 golden 值
# ============================================================================

import numpy as np
import os
import sys

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 解析命令行参数（4D shape）
# 默认 4D shape: [1, 1, 1, 128]
shape = [1, 1, 1, 128]
if len(sys.argv) >= 2:
    shape = [int(d) for d in sys.argv[1].split(',')]
# 补齐到 4 维
while len(shape) < 4:
    shape.insert(0, 1)

total_length = 1
for d in shape:
    total_length *= d

print(f"Generating test data for shape {shape}, total_length={total_length}")

# 生成定义域 (-1, 1) 内的随机 float16 数据
# 使用 float32 生成，再转为 float16
np.random.seed(42)
x_f32 = np.random.uniform(-0.95, 0.95, total_length).astype(np.float32)
x_f16 = x_f32.astype(np.float16)  # float16 数据
x_f32 = x_f16.astype(np.float32)  # 转回 float32 用于 golden 计算（精确匹配 half 范围）

x_f16.tofile("input/input_x.bin")

# 计算 golden
golden = compute_golden(x_f32)
golden.astype(np.float16).tofile("output/golden.bin")

print(f"  input/input_x.bin: shape={shape}, dtype=float16, range=[{np.min(x_f16):.4f}, {np.max(x_f16):.4f}]")
print(f"  output/golden.bin: shape={shape}, dtype=float16")
print(f"  golden range: [{np.min(golden):.4f}, {np.max(golden):.4f}]")
