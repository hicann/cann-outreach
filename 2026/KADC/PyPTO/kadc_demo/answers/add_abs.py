#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
"""
add_abs Operator: y = a + |b|

This file implements the add_abs PyPTO operator:
  y = a + absolute_value(b)

Inputs:
  - a: float32 tensor, shape [n, d]
  - b: float32 tensor, shape [n, d]
Output:
  - y: float32 tensor, shape [n, d]

Precision: atol=0.000025, rtol=0.005
Dynamic axis: n

Usage:
    python add_abs.py                        # Run all tests
    python add_abs.py --list                 # List available tests
    python add_abs.py --run_mode sim         # Run in simulation mode
    python add_abs.py add_abs::test_add_abs_basic  # Run a specific test
"""

import argparse
import os
import sys
import pypto
import torch
import numpy as np
from numpy.testing import assert_allclose


def _peek_run_mode_from_argv(default: str = "npu") -> str:
    """Read run_mode early so module-level decorators can use it."""
    for idx, arg in enumerate(sys.argv):
        if arg == "--run_mode" and idx + 1 < len(sys.argv):
            value = sys.argv[idx + 1]
            if value in ("npu", "sim"):
                return value
        if arg.startswith("--run_mode="):
            value = arg.split("=", 1)[1]
            if value in ("npu", "sim"):
                return value
    return default


global_run_mode = pypto.RunMode.NPU
if _peek_run_mode_from_argv("npu") == "sim":
    global_run_mode = pypto.RunMode.SIM


def get_device_id():
    """Get and validate TILE_FWK_DEVICE_ID from environment variable.

    Returns:
        int: The device ID if valid, None otherwise.
    """
    if 'TILE_FWK_DEVICE_ID' not in os.environ:
        print("Please set the environment variable TILE_FWK_DEVICE_ID before running:")
        print("  export TILE_FWK_DEVICE_ID=0")
        return None

    try:
        device_id = int(os.environ['TILE_FWK_DEVICE_ID'])
        return device_id
    except ValueError:
        print(f"ERROR: TILE_FWK_DEVICE_ID must be an integer, got: {os.environ['TILE_FWK_DEVICE_ID']}")
        return None


# ============================================================================
# add_abs Kernel Definition
# ============================================================================

@pypto.frontend.jit(
        runtime_options={
            "run_mode": global_run_mode,
            "runtime_debug_mode": 0,
            "compile_debug_mode": 0
        }
    )
def add_abs_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32)):
    """add_abs kernel: computes y = a + |b| element-wise.

    Args:
        a: Input tensor A of shape [n, d], dtype float32.
        b: Input tensor B of shape [n, d], dtype float32.
        out: Output tensor of shape [n, d], dtype float32.
    """
    # 请在下方添加算子核心计算逻辑
    # 设置tiling
    ...
    # y = a + |b|
    ...


# ============================================================================
# Test Cases
# ============================================================================

def test_add_abs_basic(device_id: int = None):
    """Test basic usage of add_abs operator: y = a + |b|."""
    print("=" * 60)
    print("Test: Basic Usage of add_abs Operator")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    a = torch.tensor([[1.0, -2.0], [3.0, -4.0]], dtype=dtype, device=device)
    b = torch.tensor([[-2.0, 3.0], [-4.0, 5.0]], dtype=dtype, device=device)
    # expected: a + |b| = [[1+2, -2+3], [3+4, -4+5]] = [[3, 1], [7, 1]]
    expected = a + torch.abs(b)

    out = torch.empty(a.shape, dtype=dtype, device=device)
    add_abs_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=0.005, atol=0.000025)
    print(f"Input a:    {a}")
    print(f"Input b:    {b}")
    print(f"Output:     {out}")
    print(f"Expected:   {expected}")
    print("✓ Basic usage of add_abs operator completed successfully")


def test_add_abs_dynamic_n(device_id: int = None):
    """Test add_abs operator with dynamic n-axis: different n sizes at runtime."""
    print("=" * 60)
    print("Test: add_abs Operator - Dynamic n-axis")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.float32
    d = 4

    # Test multiple values of n to verify dynamic axis support
    for n in [3, 7, 15]:
        a = torch.randn(n, d, dtype=dtype, device=device)
        b = torch.randn(n, d, dtype=dtype, device=device)
        expected = a + torch.abs(b)

        out = torch.empty(a.shape, dtype=dtype, device=device)
        add_abs_kernel(a, b, out)
        if global_run_mode == pypto.RunMode.NPU:
            assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=0.005, atol=0.000025)

        max_diff = np.abs(out.cpu().numpy() - expected.cpu().numpy()).max()
        print(f"  n={n:3d}, d={d}: max_diff={max_diff:.8f}  ✓")

    print("✓ Dynamic n-axis test completed successfully")


def test_add_abs_edge_cases(device_id: int = None):
    """Test add_abs operator with edge cases: zeros, all-negative b, large values."""
    print("=" * 60)
    print("Test: add_abs Operator - Edge Cases")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'
    dtype = torch.float32

    # 1. All zeros
    a = torch.zeros((2, 3), dtype=dtype, device=device)
    b = torch.zeros((2, 3), dtype=dtype, device=device)
    expected = a + torch.abs(b)
    out = torch.empty(a.shape, dtype=dtype, device=device)
    add_abs_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=0.005, atol=0.000025)
    print("  [zeros]   a=zeros, b=zeros     ✓")

    # 2. b all negative — abs should flip signs
    a = torch.ones((2, 3), dtype=dtype, device=device)
    b = torch.full((2, 3), -3.0, dtype=dtype, device=device)
    expected = a + torch.abs(b)  # ones + 3 = 4 everywhere
    out = torch.empty(a.shape, dtype=dtype, device=device)
    add_abs_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=0.005, atol=0.000025)
    print("  [neg_b]   a=ones,  b=-3        ✓")

    # 3. Large absolute values
    a = torch.tensor([[100.0, -200.0]], dtype=dtype, device=device)
    b = torch.tensor([[-50.0, 150.0]], dtype=dtype, device=device)
    expected = a + torch.abs(b)  # [[100+50, -200+150]] = [[150, -50]]
    out = torch.empty(a.shape, dtype=dtype, device=device)
    add_abs_kernel(a, b, out)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().numpy(), expected.cpu().numpy(), rtol=0.005, atol=0.000025)
    print("  [large]   a=[100,-200], b=[-50,150]  ✓")

    print("✓ Edge cases test completed successfully")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run add_abs operator tests.

    Usage:
        python add_abs.py                        # Run all tests
        python add_abs.py --list                 # List all available tests
        python add_abs.py add_abs::test_add_abs_basic  # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO add_abs Operator Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Run all tests
  %(prog)s --list                       List all available tests
  %(prog)s add_abs::test_add_abs_basic    Run a specific test
  %(prog)s --run_mode sim               Run in simulation mode
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs="?",
        help='Run a specific test case (e.g., add_abs::test_add_abs_basic). If omitted, run all tests.'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available tests and exit'
    )
    parser.add_argument(
        "--run_mode", "--run-mode",
        nargs="?", type=str, default="npu", choices=["npu", "sim"],
        help='Run mode: "npu" (default) or "sim".'
    )

    args = parser.parse_args()

    # Registry of all test cases
    examples = {
        'add_abs::test_add_abs_basic': {
            'name': 'Basic usage of add_abs operator',
            'description': 'Verify y = a + |b| with a fixed small tensor.',
            'function': test_add_abs_basic
        },
        'add_abs::test_add_abs_dynamic_n': {
            'name': 'Dynamic n-axis test',
            'description': 'Verify the operator handles different n sizes at runtime.',
            'function': test_add_abs_dynamic_n
        },
        'add_abs::test_add_abs_edge_cases': {
            'name': 'Edge cases (zeros, all-negative, large values)',
            'description': 'Verify correctness on boundary inputs.',
            'function': test_add_abs_edge_cases
        },
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Tests for add_abs Operator")
        print("=" * 60 + "\n")
        for case_key, ex_info in sorted(examples.items()):
            print(f"  {case_key}")
            print(f"     Name: {ex_info['name']}")
            print(f"     Description: {ex_info['description']}\n")
        return

    # Select tests to run
    if args.example_id:
        if args.example_id not in examples:
            print(f"ERROR: Invalid case '{args.example_id}'")
            print(f"Valid cases are: {', '.join(sorted(examples.keys()))}")
            print("\nUse --list to see all available tests.")
            sys.exit(1)
        examples_to_run = [(args.example_id, examples[args.example_id])]
    else:
        examples_to_run = list(examples.items())

    print("\n" + "=" * 60)
    print("PyPTO add_abs Operator Tests")
    print("=" * 60 + "\n")

    device_id = None
    if args.run_mode == "npu":
        device_id = get_device_id()
        if device_id is None:
            return
        import torch_npu
        torch.npu.set_device(device_id)

    try:
        for case_key, ex_info in examples_to_run:
            ex_info['function'](device_id)

        if len(examples_to_run) > 1:
            print("=" * 60)
            print("All add_abs tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
