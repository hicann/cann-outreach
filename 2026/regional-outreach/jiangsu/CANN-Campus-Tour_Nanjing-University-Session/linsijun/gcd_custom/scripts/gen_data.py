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
# 测试数据生成脚本 - Gcd Float16
# ============================================================================
#
# 生成多组 4D broadcast 测试数据:
#   [A] self=[1,1,1,128],  other=[1,1,1,128]  → out=[1,1,1,128]
#   [B] self=[1,1,1,128],  other=[1,1,4,1]    → out=[1,1,4,128]
#   [C] self=[2,1,1,128],  other=[1,4,1,1]    → out=[2,4,1,128]
# ============================================================================

import numpy as np
import os
import argparse

from golden import compute_golden

# NumPy broadcast 工具函数
def broadcast_expand_4d(data, src_shape, dst_shape):
    """将 4D 数据从 src_shape broadcast-expand 到 dst_shape"""
    # 创建原始 shape 的 view，利用 NumPy 自动 broadcast
    src_reshaped = data.reshape(src_shape)
    # 利用 np.broadcast_to 自动扩展
    expanded = np.broadcast_to(src_reshaped, dst_shape)
    return expanded


# 测试用例定义 (4D shapes)
TEST_CASES = {
    "A": {
        "self_shape":  (1, 1, 1, 128),
        "other_shape": (1, 1, 1, 128),
        "desc": "Simple: same shape"
    },
    "B": {
        "self_shape":  (1, 1, 1, 128),
        "other_shape": (1, 1, 4, 1),
        "desc": "Broadcast on N2 (other N2=4, N1=1)"
    },
    "C": {
        "self_shape":  (2, 1, 1, 128),
        "other_shape": (1, 4, 1, 1),
        "desc": "Broadcast on N1+N3 (self N0=2, other N1=4)"
    },
}

def main():
    parser = argparse.ArgumentParser(description="Generate test data for gcd_float16")
    parser.add_argument("--case", choices=list(TEST_CASES.keys()) + ["all"],
                        default="all", help="Test case to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    cases_to_gen = list(TEST_CASES.items()) if args.case == "all" else [(args.case, TEST_CASES[args.case])]

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    for name, cfg in cases_to_gen:
        self_shape = cfg["self_shape"]
        other_shape = cfg["other_shape"]
        
        # 计算 broadcast 输出 shape
        out_shape = tuple(max(s, o) for s, o in zip(self_shape, other_shape))
        self_size = int(np.prod(self_shape))
        other_size = int(np.prod(other_shape))
        out_size = int(np.prod(out_shape))

        print(f"\n=== Test Case {name}: {cfg['desc']} ===")
        print(f"  self:  {self_shape}  ({self_size} elements)")
        print(f"  other: {other_shape} ({other_size} elements)")
        print(f"  out:   {out_shape}  ({out_size} elements)")

        # 生成原始数据 (接近整数的 float16 值)
        self_data = np.random.randint(1, 101, size=self_size).astype(np.float16)
        other_data = np.random.randint(1, 101, size=other_size).astype(np.float16)

        # 混入负值
        neg_mask = np.random.random(self_size) < 0.33
        self_data[neg_mask] *= -1
        neg_mask = np.random.random(other_size) < 0.33
        other_data[neg_mask] *= -1

        # 混入固定边界值
        self_flat = self_data.reshape(-1)
        other_flat = other_data.reshape(-1)
        if self_size > 0:
            self_flat[0] = np.float16(0.0)      # gcd(0, other) = |other|
        if other_size > 0:
            other_flat[0] = np.float16(12.0)
        if self_size > 1:
            self_flat[1] = np.float16(48.0)     # gcd(48, 36) = 12
        if other_size > 1:
            other_flat[1] = np.float16(36.0)
        if self_size > 2:
            self_flat[2] = np.float16(17.0)     # gcd(17, 1) = 1
        if other_size > 2:
            other_flat[2] = np.float16(1.0)
        if self_size > 3:
            self_flat[3] = np.float16(7.5)      # gcd(7.5, 2.5) = 2.5
        if other_size > 3:
            other_flat[3] = np.float16(2.5)

        # 保存原始数据 (host 端用于 broadcast expand 后再喂入 kernel)
        case_dir = f"input/{name}"
        os.makedirs(case_dir, exist_ok=True)
        os.makedirs(f"output/{name}", exist_ok=True)

        self_path = f"{case_dir}/self.bin"
        other_path = f"{case_dir}/other.bin"
        self_data.tofile(self_path)
        other_data.tofile(other_path)

        # 计算 broadcast 扩展后的 golden
        self_reshaped = self_data.reshape(self_shape)
        other_reshaped = other_data.reshape(other_shape)

        # NumPy broadcast 两个输入
        self_bc = np.broadcast_to(self_reshaped, out_shape)
        other_bc = np.broadcast_to(other_reshaped, out_shape)

        golden = compute_golden(self_bc, other_bc)
        golden_path = f"output/{name}/golden.bin"
        golden.tofile(golden_path)

        print(f"  saved: {self_path}, {other_path}, {golden_path}")
        print(f"  golden dtype={golden.dtype}, shape={golden.shape}")


if __name__ == "__main__":
    main()
