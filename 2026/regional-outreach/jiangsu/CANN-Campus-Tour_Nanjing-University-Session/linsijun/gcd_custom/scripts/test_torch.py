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
# PyTorch 通路测试脚本 - Gcd Float16
# ============================================================================
#
# 测试用例涵盖:
#   1. 相同 shape 的 4D 输入
#   2. Broadcast 场景 (self.shape != other.shape)
#   3. 零值、边界值、负值
#   4. 浮点值 (非整数)
# ============================================================================

import sys
import os

import torch
import torch_npu
import numpy as np

from golden import compute_golden

# 算子配置
SO_NAME = "libgcd_float16_ops.so"   # .so 文件名
OP_NAME = "gcd_float16"             # 算子名称
DTYPE = torch.float16               # 数据类型
ATOL = 1e-3
RTOL = 1e-3

# 测试 shape (4D)
TEST_SHAPES = [
    # (self_shape, other_shape, out_shape)
    ((1, 1, 1, 128),   (1, 1, 1, 128),   (1, 1, 1, 128),   "same shape"),
    ((1, 1, 1, 128),   (1, 1, 4, 1),     (1, 1, 4, 128),   "N2 broadcast"),
    ((2, 1, 1, 128),   (1, 4, 1, 1),     (2, 4, 1, 128),   "N1+N3 broadcast"),
]


def run_test(name, self_t, other_t):
    """运行单个测试用例，返回 (name, passed, max_diff)"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    out_t = op_fn(self_t.npu(), other_t.npu())

    # golden 计算需要 numpy 或 torch 层面已完成 broadcast
    # PyTorch 的 custom op 会自动处理 broadcast
    # golden: broadcast 后的计算结果
    self_np = self_t.cpu().numpy()
    other_np = other_t.cpu().numpy()

    # 手动 broadcast (用于 golden 计算)
    self_bc = np.broadcast_to(self_np, self_np.shape)  # same if already broadcast
    other_bc = np.broadcast_to(other_np, other_np.shape)

    # 实际 golden 使用已 broadcast 的输入
    golden_np = compute_golden(self_t.npu().float().cpu(), other_t.npu().float().cpu())
    # ^ 上面这样不对，golden 应该基于 broadcast 后的输入

    # 直接用原始 numpy 计算 golden (让 golden.py 处理 broadcast 后的输入)
    # 但 test_torch.py 的输入已确保 self 和 other 同 shape
    golden_np = compute_golden(self_np, other_np)
    golden_t = torch.from_numpy(golden_np).npu()

    max_diff = torch.max(torch.abs(out_t - golden_t)).item()
    passed = torch.allclose(out_t.cpu(), golden_t.cpu(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def run_broadcast_test(name, self_shape, other_shape, out_shape):
    """运行 broadcast 测试: self 和 other shape 不同"""
    self_t = torch.randn(*self_shape, dtype=DTYPE)
    other_t = torch.randn(*other_shape, dtype=DTYPE)

    # 填入一些已知值确保可验证
    self_t[0, 0, 0, 0] = 48.0
    other_t[0, 0, 0, 0] = 36.0  # gcd = 12

    # 通过 torch broadcast 机制扩展
    self_bc, other_bc = torch.broadcast_tensors(self_t, other_t)
    out_shape_computed = self_bc.shape
    assert out_shape_computed == out_shape, f"Shape mismatch: {out_shape_computed} != {out_shape}"

    # 调用算子
    op_fn = getattr(torch.ops.npu, OP_NAME)
    out_t = op_fn(self_t.npu(), other_t.npu())

    # golden
    golden_np = compute_golden(self_bc.numpy(), other_bc.numpy())
    golden_t = torch.from_numpy(golden_np)

    max_diff = torch.max(torch.abs(out_t.cpu() - golden_t)).item()
    passed = torch.allclose(out_t.cpu(), golden_t, atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    # 加载算子库
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # === 相同 shape 测试 ===
    for self_shape, other_shape, out_shape, desc in TEST_SHAPES:
        if self_shape == other_shape:
            # 相同 shape: 用 run_test
            self_t = torch.randn(*self_shape, dtype=DTYPE)
            other_t = torch.randn(*other_shape, dtype=DTYPE)
            # 覆盖已知值
            self_t[0, 0, 0, 0] = 0.0
            other_t[0, 0, 0, 0] = 12.0   # gcd(0,12) = 12
            if self_shape[3] > 1:
                self_t[0, 0, 0, 1] = 48.0
                other_t[0, 0, 0, 1] = 36.0   # gcd = 12
            if self_shape[3] > 2:
                self_t[0, 0, 0, 2] = -17.0
                other_t[0, 0, 0, 2] = 1.0    # gcd = 1
            if self_shape[3] > 3:
                self_t[0, 0, 0, 3] = 7.5
                other_t[0, 0, 0, 3] = 2.5    # gcd = 2.5

            results.append(run_test(f"P1 same_shape {desc}", self_t, other_t))

    # === Broadcast 测试 ===
    for self_shape, other_shape, out_shape, desc in TEST_SHAPES:
        if self_shape != other_shape:
            self_t = torch.randn(*self_shape, dtype=DTYPE)
            other_t = torch.randn(*other_shape, dtype=DTYPE)

            self_bc, other_bc = torch.broadcast_tensors(self_t, other_t)

            results.append(run_broadcast_test(
                f"P2 broadcast {desc}",
                self_t, other_t, out_shape))

    # === 零值和边界值 ===
    for shape_desc, shape in [("small", (1, 1, 1, 128))]:
        self_t = torch.zeros(*shape, dtype=DTYPE)
        other_t = torch.full(shape, 1.0, dtype=DTYPE) * 24.0  # gcd(0, 24) = 24
        results.append(run_test(f"P3 zeros {shape_desc}", self_t, other_t))

    # === 全负值 ===
    for shape_desc, shape in [("small", (1, 1, 1, 128))]:
        self_t = -torch.randint(1, 51, shape, dtype=DTYPE)
        other_t = -torch.randint(1, 51, shape, dtype=DTYPE)
        results.append(run_test(f"P4 all_neg {shape_desc}", self_t, other_t))

    # === 大值 ===
    for shape_desc, shape in [("small", (1, 1, 1, 128))]:
        self_t = torch.randint(1000, 60000, shape, dtype=DTYPE)
        other_t = torch.randint(1000, 60000, shape, dtype=DTYPE)
        results.append(run_test(f"P5 large_vals {shape_desc}", self_t, other_t))

    # 汇总
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*60}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff:.6f})")
    print(f"{'='*60}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
