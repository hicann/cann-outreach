# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================================
# PyTorch 通路测试脚本 — Gcd 算子 (float16, 4D broadcast)
# ============================================================================
#
# 测试 broadcast: self=[N4,N3,N2,N1], other=[M4,M3,M2,M1] → broadcast broadcast
# 覆盖 shape: [1,1,1,1], [2,3,8,16], [4,1,8,16], [1,3,1,1] 等组合
# ============================================================================

import sys
import os

import torch
import torch_npu
import numpy as np

from golden import compute_golden

# 算子配置
SO_NAME = "libgcd_custom_ops.so"    # .so 文件名
OP_NAME = "gcd_custom"              # 算子名称
DTYPE = torch.float16
ATOL = 1e-4
RTOL = 1e-2


def run_test(name, self_cpu, other_cpu):
    """运行单个测试用例"""
    op_fn = getattr(torch.ops.npu, OP_NAME)
    out = op_fn(self_cpu.npu(), other_cpu.npu())

    # compute_golden 接收 numpy array
    golden = compute_golden(self_cpu.numpy(), other_cpu.numpy())
    golden_t = torch.from_numpy(golden).npu()

    max_diff = torch.max(torch.abs(out - golden_t)).item()
    passed = torch.allclose(out.cpu(), golden_t.cpu(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # P1: 4D 完全 broadcast — self=[2,1,8,16], other=[2,3,1,1]
    self_cpu = torch.randn(2, 1, 8, 16, dtype=DTYPE)
    other_cpu = torch.randn(2, 3, 1, 1, dtype=DTYPE)
    results.append(run_test("P1 4D broadcast [2,1,8,16] vs [2,3,1,1]", self_cpu, other_cpu))

    # P2: 4D 自 broadcast — self=[4,6,16,32], other=[4,6,16,32] (同 shape)
    self_cpu = torch.randn(4, 6, 16, 32, dtype=DTYPE)
    other_cpu = torch.randn(4, 6, 16, 32, dtype=DTYPE)
    results.append(run_test("P2 same shape [4,6,16,32]", self_cpu, other_cpu))

    # P3: 4D 全维度 broadcast — self=[1,1,1,1], other=[2,3,8,16]
    self_cpu = torch.randn(1, 1, 1, 1, dtype=DTYPE)
    other_cpu = torch.randn(2, 3, 8, 16, dtype=DTYPE)
    results.append(run_test("P3 scalar broadcast [1,1,1,1] vs [2,3,8,16]", self_cpu, other_cpu))

    # P4: 零值测试 — self 含零, other 含零
    self_cpu = torch.zeros(2, 3, 8, 16, dtype=DTYPE)
    other_cpu = torch.full((2, 3, 8, 16), 2.0, dtype=DTYPE)
    results.append(run_test("P4 gcd(0, 2.0)", self_cpu, other_cpu))

    # P5: 半零测试 — gcd(x, 0) = |x|
    self_cpu = torch.randn(2, 3, 8, 16, dtype=DTYPE)
    other_cpu = torch.zeros(2, 3, 8, 16, dtype=DTYPE)
    results.append(run_test("P5 gcd(x, 0)", self_cpu, other_cpu))

    # P6: 正负混合 — 取绝对值后计算
    self_cpu = torch.randn(2, 3, 8, 16, dtype=DTYPE)
    other_cpu = -torch.abs(torch.randn(2, 3, 8, 16, dtype=DTYPE))
    results.append(run_test("P6 pos/neg mix", self_cpu, other_cpu))

    # P7: 全相同值 — gcd(a, a) = a
    val = torch.tensor(3.0, dtype=DTYPE)
    self_cpu = val.expand(4, 1, 8, 16)
    other_cpu = val.expand(4, 6, 8, 16)  # broadcast: [4,6,8,16]
    results.append(run_test("P7 gcd(a, a) no dim3 BC [4,1,8,16] vs [4,6,8,16]", self_cpu, other_cpu))

    # 汇总
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*50}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff:.6e})")
    print(f"{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
