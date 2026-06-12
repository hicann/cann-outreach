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
# PyTorch 通路测试脚本 - Atanh 算子（float16, 4D ND）
# ============================================================================

import sys
import os

import torch
import torch_npu
import numpy as np

# 添加 scripts 目录到 sys.path 以便导入 golden.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import compute_golden

SO_NAME = "libatanh_custom_ops.so"
OP_NAME = "atanh_custom"
DTYPE = torch.float16
ATOL = 1e-3
RTOL = 1e-3

# 4D shape
SHAPES = [
    (1, 4, 16, 64),        # 4096
    (2, 8, 32, 128),       # 65536
    (4, 16, 64, 256),      # 1M
]


def run_test(name, x):
    """运行单个测试用例"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(x.npu())
    golden = compute_golden(x).npu()
    max_diff = torch.max(torch.abs(y - golden)).item()
    passed = torch.allclose(y.cpu(), golden.cpu(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    so_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # P1-P3: 三种 4D shape 随机数据
    for idx, shape in enumerate(SHAPES):
        # 使用 tanh 确保值在 (-1, 1)
        x = torch.tanh(torch.randn(shape, dtype=torch.float32)).to(DTYPE)
        results.append(run_test(f"P{idx+1} random shape={shape}", x))

    # P4: 零值
    x = torch.zeros(SHAPES[0], dtype=DTYPE)
    results.append(run_test("P4 zeros", x))

    # P5: 正数
    x = torch.tanh(torch.abs(torch.randn(SHAPES[0], dtype=torch.float32))).to(DTYPE)
    results.append(run_test("P5 positive", x))

    # P6: 负数
    x = torch.tanh(-torch.abs(torch.randn(SHAPES[0], dtype=torch.float32))).to(DTYPE)
    results.append(run_test("P6 negative", x))

    # P7: 接近边界值
    x_batch = torch.tensor([[0.5, -0.5, 0.999, -0.999]], dtype=DTYPE)
    x_pad = torch.zeros(SHAPES[0], dtype=DTYPE)
    for i in range(min(x_batch.shape[1], x_pad.numel())):
        x_pad.flatten()[i] = x_batch.flatten()[i]
    results.append(run_test("P7 near boundary", x_pad))

    # 汇总
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*50}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff})")
    print(f"{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
