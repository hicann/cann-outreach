# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# ============================================================================
# 测试数据生成脚本 — Gcd 算子 (float16, 4D broadcast)
# ============================================================================
#
# 4D broadcast 示例: self=[2,1,8,16], other=[2,3,1,1] → output=[2,3,8,16]
# 修改 __main__ 中的 shape 即可切换测试数据。
# ============================================================================

import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# [CONFIG] 4D shape 配置
SELF_SHAPE = (2, 1, 8, 16)     # 需满足: other 在 dim=1 和 dim=3 上 broadcast
OTHER_SHAPE = (2, 3, 1, 1)     # 与 self broadcast 后: (2, 3, 8, 16)
BC_SHAPE = (2, 3, 8, 16)       # broadcast 后的公共 shape

dtype = np.float16
total_length = int(np.prod(BC_SHAPE))  # 2*3*8*16 = 768

def main():
    # 生成输入数据
    np.random.seed(42)
    self_data = np.random.randn(*SELF_SHAPE).astype(np.float32).astype(dtype)
    other_data = np.random.randn(*OTHER_SHAPE).astype(np.float32).astype(dtype)

    # 添加边界值覆盖
    self_data[0, 0, 0, 0] = np.float16(0.0)       # 零值
    other_data[0, 0, 0, 0] = np.float16(0.0)      # gcd(0,0)=0

    # 部分 other 元素设为零 (测试 gcd(x,0)=|x|)
    self_data[1, 0, 0, 0] = np.float16(3.14)
    other_data[1, 0, 0, 0] = np.float16(0.0)

    # Broadcast 展开到公共 shape
    self_bc = np.broadcast_to(self_data, BC_SHAPE)
    other_bc = np.broadcast_to(other_data, BC_SHAPE)

    # 展平为一维写入文件
    self_bc.flatten().tofile("input/input_self.bin")
    other_bc.flatten().tofile("input/input_other.bin")

    # 计算 golden
    golden = compute_golden(self_bc, other_bc)
    golden.tofile("output/golden.bin")

    print(f"Generated GCD test data:")
    print(f"  self shape:  {SELF_SHAPE}")
    print(f"  other shape: {OTHER_SHAPE}")
    print(f"  bc shape:    {BC_SHAPE}")
    print(f"  total elements: {total_length}")
    print(f"  dtype: {dtype}")
    print(f"  input/input_self.bin:  {self_bc.shape}, {self_bc.dtype}")
    print(f"  input/input_other.bin: {other_bc.shape}, {other_bc.dtype}")
    print(f"  golden range: [{np.min(golden)}, {np.max(golden)}]")


if __name__ == "__main__":
    main()
