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
# 测试数据生成脚本 - Relu 算子
# ============================================================================
#
# ReLU: z[i] = max(0, x[i])
# 生成包含正、负、零混合数据，三个 4D Shape:
#   - [1, 1, 1, 128]     → 128   元素
#   - [1, 4, 32, 64]    → 8192  元素
#   - [8, 16, 32, 32]   → 131072 元素
# ============================================================================

import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

SHAPES = {
    "1_1_1_128":   (1, 1, 1, 128),
    "1_4_32_64":   (1, 4, 32, 64),
    "8_16_32_32":  (8, 16, 32, 32),
}

for name, shape in SHAPES.items():
    # 生成正负混合数据 (范围 -10 ~ 10)
    x = np.random.uniform(-10.0, 10.0, size=shape).astype(np.float16)
    x.tofile(f"input/input_x_{name}.bin")

    golden = compute_golden(x)
    golden.tofile(f"output/golden_{name}.bin")

    npos = np.sum(golden > 0)
    nneg = np.sum(x < 0)
    nzero = np.sum(x == 0)
    print(f"  [{name}] shape={shape}, elements={x.size}, "
          f"positive={npos}, neg_input={nneg}, zero_input={nzero}")
