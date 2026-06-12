# ============================================================================
# PyTorch 通路测试脚本 - ReLU 算子（float16, 4D ND）
# ============================================================================
#
# 测试多种 4D Shape + 边界值
# ============================================================================

import sys
import os

import torch
import torch_npu

from golden import compute_golden

SO_NAME = "librelu_custom_ops.so"
OP_NAME = "relu_custom"
DTYPE = torch.float16
ATOL = 1e-6
RTOL = 1e-4


def run_test(name, x):
    op_fn = getattr(torch.ops.npu, OP_NAME)
    y = op_fn(x.npu())
    golden = compute_golden(x).npu()
    max_diff = torch.max(torch.abs(y - golden)).item()
    passed = torch.allclose(y.cpu().float(), golden.cpu().float(), atol=ATOL, rtol=RTOL)
    return name, passed, max_diff


def main():
    so_path = os.path.join("build", SO_NAME)
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run 'cmake .. && make' first.")
        sys.exit(1)
    torch.ops.load_library(so_path)

    results = []

    # T1~T4: 多种 4D Shape
    shapes = [
        ("T1 [1,1,1,128]",       (1, 1, 1, 128)),
        ("T2 [1,4,16,128]",      (1, 4, 16, 128)),
        ("T3 [2,8,32,64]",       (2, 8, 32, 64)),
        ("T4 [4,16,64,128]",     (4, 16, 64, 128)),
    ]
    for name, shape in shapes:
        x = torch.randn(shape, dtype=DTYPE) * 3.0  # 正负混合
        results.append(run_test(name, x))

    # P1: 全正
    x = torch.abs(torch.randn(2, 4, 8, 16, dtype=DTYPE)) + 0.1
    results.append(run_test("P1 all_pos", x))

    # P2: 全负
    x = -torch.abs(torch.randn(2, 4, 8, 16, dtype=DTYPE)) - 0.1
    results.append(run_test("P2 all_neg", x))

    # P3: 全零
    x = torch.zeros(2, 4, 8, 16, dtype=DTYPE)
    results.append(run_test("P3 zeros", x))

    # P4: 大正数和大负数混合
    x = torch.tensor([[[[65504.0, -65504.0, 0.0, -0.0]]]], dtype=DTYPE)
    results.append(run_test("P4 boundary", x))

    # 汇总
    total = len(results)
    passed = sum(r[1] for r in results)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"PyTorch test results ({OP_NAME})")
    print(f"{'='*50}")
    for name, ok, diff in results:
        print(f"  {name}: {'PASSED' if ok else 'FAILED'} (Max diff={diff:.6f})")
    print(f"{'='*50}")
    print(f"Total: {total}, Passed: {passed}, Failed: {failed}")
    print(f"Status: {'PASSED' if failed == 0 else 'FAILED'}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
