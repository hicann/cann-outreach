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
# PyTorch 通路测试脚本 - Relu 算子
# ============================================================================

import sys
import os

import torch
import torch_npu

from golden import compute_golden

SO_NAME = "librelu_custom_ops.so"
OP_NAME = "relu_custom"
DTYPE = torch.float16
ATOL = 1e-4
RTOL = 1e-3

SHAPES = [
    ("S1_128",   (1, 1, 1, 128)),
    ("S2_4K",    (1, 4, 32, 64)),
    ("S3_128K",  (8, 16, 32, 32)),
]


def run_test(name, x):
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(x.npu())
    golden_np = compute_golden(x.cpu().numpy())
    golden = torch.from_numpy(golden_np)
    max_diff = torch.max(torch.abs(y.cpu().float() - golden.float())).item()
    passed = torch.allclose(y.cpu().float(), golden.float(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    for name, shape in SHAPES:
        # P1: 正负混合
        x = (torch.rand(*shape, dtype=torch.float32) * 20 - 10).to(DTYPE)
        results.append(run_test(f"{name} pos_neg_mix", x))

        # P2: 全正
        x = (torch.rand(*shape, dtype=torch.float32) * 10).to(DTYPE)
        results.append(run_test(f"{name} all_pos", x))

        # P3: 全负
        x = (-torch.rand(*shape, dtype=torch.float32) * 10).to(DTYPE)
        results.append(run_test(f"{name} all_neg", x))

        # P4: 零值
        x = torch.zeros(shape, dtype=DTYPE)
        results.append(run_test(f"{name} zeros", x))

        # P5: 大正数边界
        x = torch.full(shape, 1000.0, dtype=DTYPE)
        results.append(run_test(f"{name} large_pos", x))

    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'=' * 50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'=' * 50}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff:.6f})")
    print(f"{'=' * 50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
