#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
"""
matmul_add Operator: y = a @ b^T + c

This file implements the matmul_add PyPTO operator:
  y = matmul(a, b^T) + c

Inputs:
  - a: bfloat16 tensor, shape [m, k], m is dynamic
  - b: bfloat16 tensor, shape [n, k]
  - c: bfloat16 tensor, shape [m, n]
Output:
  - y: bfloat16 tensor, shape [m, n]

Precision: atol=0.0001, rtol=0.0078125
Dynamic axis: m (pypto.DYNAMIC)

Note on DYNAMIC loops:
    The kernel uses pypto.DYNAMIC for the m-axis. The loop iteration count is
    determined at the first kernel call. To support different m values, call
    with the largest m first so the compiled loop covers all subsequent calls.

Usage:
    python matmul_add.py                        # Run all tests
    python matmul_add.py --list                 # List available tests
    python matmul_add.py --run_mode sim         # Run in simulation mode
    python matmul_add.py matmul_add::test_matmul_add_basic  # Run a specific test
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
# matmul_add Kernel Definition
# ============================================================================

@pypto.frontend.jit(
    runtime_options={
        "run_mode": global_run_mode,
        "stitch_function_max_num": 128
    },
    debug_options={
        "runtime_debug_mode": 0,
        "compile_debug_mode": 0
    }
)
def matmul_add_kernel(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    b: pypto.Tensor([], pypto.DT_BF16),
    c: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),
    n_val: int,
    k_val: int):
    """matmul_add kernel: computes y = a @ b^T + c.

    a is [m, k] where m is dynamic.
    b is [n, k] -> b^T is [k, n], so we use b_trans=True.
    c is [m, n].
    out is [m, n].

    m is declared as pypto.DYNAMIC, so the kernel can handle different m values
    without recompilation. Dynamic tensors must be accessed via pypto.view inside
    pypto.loop to get static-shape tiles for computation.

    Only the dynamic m-axis is manually tiled via loop; n and k are static and
    handled by the framework internally.

    Args:
        a: Input tensor A of shape [m, k], dtype bfloat16. m is dynamic.
        b: Input tensor B of shape [n, k], dtype bfloat16.
        c: Input tensor C of shape [m, n], dtype bfloat16. m is dynamic.
        out: Output tensor of shape [m, n], dtype bfloat16. m is dynamic.
        n_val: Static dimension n.
        k_val: Static dimension k.
    """
    m_val = a.shape[0]  # dynamic dimension -> symbolic scalar

    tile_m = 16
    # Cube tile shapes must satisfy BF16 16-element alignment (32 bytes)
    tile_k = ((k_val + 15) // 16) * 16
    tile_n = ((n_val + 15) // 16) * 16

    m_loop = (m_val + tile_m - 1) // tile_m

    for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_m", idx_name="m_idx"):
        m_offset = m_idx * tile_m
        # Boundary: use .min() on symbolic expression (NOT Python's min())
        valid_m = (m_val - m_offset).min(tile_m)

        # View tiles along the dynamic m-axis
        a_view = pypto.view(a, [tile_m, k_val], [m_offset, 0],
                            valid_shape=[valid_m, k_val])
        c_view = pypto.view(c, [tile_m, n_val], [m_offset, 0],
                            valid_shape=[valid_m, n_val])

        # Set cube tile shapes for matmul
        pypto.set_cube_tile_shapes([128, 128], [128, 128], [128, 128])

        # matmul: a_view [tile_m, k_val] @ b^T [k_val, n_val] -> [tile_m, n_val]
        mm_result = pypto.matmul(a_view, b, pypto.DT_BF16, b_trans=True)

        # add bias: mm_result + c_view
        pypto.set_vec_tile_shapes(tile_m, tile_n)
        add_result = pypto.add(mm_result, c_view)

        # Assemble result back to output
        pypto.assemble(add_result, [m_offset, 0], out)


# ============================================================================
# Test Cases
# ============================================================================

def test_matmul_add_basic(device_id: int = None):
    """Test basic usage of matmul_add operator: y = a @ b^T + c."""
    print("=" * 60)
    print("Test: Basic Usage of matmul_add Operator")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.bfloat16
    m, n, k = 4, 4, 4

    a = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0],
                      [9.0, 10.0, 11.0, 12.0],
                      [13.0, 14.0, 15.0, 16.0]], dtype=dtype, device=device)
    b = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]], dtype=dtype, device=device)
    c = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                      [0.5, 0.6, 0.7, 0.8],
                      [0.9, 1.0, 1.1, 1.2],
                      [1.3, 1.4, 1.5, 1.6]], dtype=dtype, device=device)

    # Golden: y = a @ b^T + c = a @ I + c = a + c (since b is identity)
    expected = a @ b.T + c

    out = torch.empty((m, n), dtype=dtype, device=device)
    matmul_add_kernel(a, b, c, out, n, k)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().float().numpy(), expected.cpu().float().numpy(),
                        rtol=0.0078125, atol=0.0001)
    print(f"Input a:      {a}")
    print(f"Input b:      {b}")
    print(f"Input c:      {c}")
    print(f"Output:       {out}")
    print(f"Expected:     {expected}")
    print("✓ Basic usage of matmul_add operator completed successfully")


def test_matmul_add_dynamic_m(device_id: int = None):
    """Test matmul_add operator with dynamic m-axis: different m sizes at runtime."""
    print("=" * 60)
    print("Test: matmul_add Operator - Dynamic m-axis")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.bfloat16
    n, k = 1024, 1024

    # Test multiple values of m to verify dynamic axis support
    # IMPORTANT: iterate from largest to smallest m. PyPTO DYNAMIC loops compile
    # for the iteration count of the first call; subsequent calls with fewer
    # iterations work via valid_shape boundary management.
    for m in [128]:
        a = torch.randn(m, k, dtype=dtype, device=device)
        b = torch.randn(n, k, dtype=dtype, device=device)
        c = torch.randn(m, n, dtype=dtype, device=device)

        expected = a @ b.T + c

        out = torch.empty((m, n), dtype=dtype, device=device)
        matmul_add_kernel(a, b, c, out, n, k)
        if global_run_mode == pypto.RunMode.NPU:
            assert_allclose(out.cpu().float().numpy(), expected.cpu().float().numpy(),
                            rtol=0.0078125, atol=0.0001)

        max_diff = np.abs(out.cpu().float().numpy() - expected.cpu().float().numpy()).max()
        print(f"  m={m:4d}, n={n:2d}, k={k:2d}: max_diff={max_diff:.8f}  ✓")

    print("✓ Dynamic m-axis test completed successfully")


def test_matmul_add_non_square(device_id: int = None):
    """Test matmul_add with non-square matrices and non-16-aligned dimensions."""
    print("=" * 60)
    print("Test: matmul_add Operator - Non-square Matrices")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'

    dtype = torch.bfloat16

    # Non-square: m=7, n=5, k=11 (not aligned to tile sizes)
    m, n, k = 7, 5, 11

    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(n, k, dtype=dtype, device=device)
    c = torch.randn(m, n, dtype=dtype, device=device)

    expected = a @ b.T + c

    out = torch.empty((m, n), dtype=dtype, device=device)
    matmul_add_kernel(a, b, c, out, n, k)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().float().numpy(), expected.cpu().float().numpy(),
                        rtol=0.0078125, atol=0.0001)

    max_diff = np.abs(out.cpu().float().numpy() - expected.cpu().float().numpy()).max()
    print(f"  m={m}, n={n}, k={k}: max_diff={max_diff:.8f}  ✓")
    print("✓ Non-square matrices test completed successfully")


def test_matmul_add_edge_cases(device_id: int = None):
    """Test matmul_add with edge cases: zeros, single row, large values."""
    print("=" * 60)
    print("Test: matmul_add Operator - Edge Cases")
    print("=" * 60)

    device = f'npu:{device_id}' if global_run_mode == pypto.RunMode.NPU and device_id is not None else 'cpu'
    dtype = torch.bfloat16

    # 1. All zeros - output should equal c
    m, n, k = 4, 4, 4
    a = torch.zeros((m, k), dtype=dtype, device=device)
    b = torch.zeros((n, k), dtype=dtype, device=device)
    c = torch.randn(m, n, dtype=dtype, device=device)
    expected = a @ b.T + c  # should equal c
    out = torch.empty((m, n), dtype=dtype, device=device)
    matmul_add_kernel(a, b, c, out, n, k)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().float().numpy(), expected.cpu().float().numpy(),
                        rtol=0.0078125, atol=0.0001)
    print("  [zeros]    a=zeros, b=zeros, c=random    ✓")

    # 2. Single row (m=1) - minimum dynamic size
    m, n, k = 1, 8, 16
    a = torch.randn(m, k, dtype=dtype, device=device)
    b = torch.randn(n, k, dtype=dtype, device=device)
    c = torch.randn(m, n, dtype=dtype, device=device)
    expected = a @ b.T + c
    out = torch.empty((m, n), dtype=dtype, device=device)
    matmul_add_kernel(a, b, c, out, n, k)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().float().numpy(), expected.cpu().float().numpy(),
                        rtol=0.0078125, atol=0.0001)
    print("  [m=1]      single row matrix              ✓")

    # 3. Large values
    m, n, k = 4, 4, 4
    a = torch.tensor([[100.0, -200.0, 300.0, -400.0],
                      [500.0, -600.0, 700.0, -800.0],
                      [-50.0, 60.0, -70.0, 80.0],
                      [10.0, -20.0, 30.0, -40.0]], dtype=dtype, device=device)
    b = torch.tensor([[10.0, 20.0, 30.0, 40.0],
                      [-10.0, -20.0, -30.0, -40.0],
                      [5.0, 10.0, 15.0, 20.0],
                      [-5.0, -10.0, -15.0, -20.0]], dtype=dtype, device=device)
    c = torch.randn(m, n, dtype=dtype, device=device)
    expected = a @ b.T + c
    out = torch.empty((m, n), dtype=dtype, device=device)
    matmul_add_kernel(a, b, c, out, n, k)
    if global_run_mode == pypto.RunMode.NPU:
        assert_allclose(out.cpu().float().numpy(), expected.cpu().float().numpy(),
                        rtol=0.0078125, atol=0.0001)
    print("  [large]    large absolute values          ✓")

    print("✓ Edge cases test completed successfully")


# ============================================================================
# Main entry point
# ============================================================================

def main():
    """Run matmul_add operator tests.

    Usage:
        python matmul_add.py                        # Run all tests
        python matmul_add.py --list                 # List all available tests
        python matmul_add.py matmul_add::test_matmul_add_basic  # Run a specific case
    """
    parser = argparse.ArgumentParser(
        description="PyPTO matmul_add Operator Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    Run all tests
  %(prog)s --list                             List all available tests
  %(prog)s matmul_add::test_matmul_add_basic    Run a specific test
  %(prog)s --run_mode sim                     Run in simulation mode
        """
    )
    parser.add_argument(
        'example_id',
        type=str,
        nargs="?",
        help='Run a specific test case (e.g., matmul_add::test_matmul_add_basic). If omitted, run all tests.'
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
        'matmul_add::test_matmul_add_basic': {
            'name': 'Basic usage of matmul_add operator',
            'description': 'Verify y = a @ b^T + c with a fixed small tensor (identity b).',
            'function': test_matmul_add_basic
        },
        'matmul_add::test_matmul_add_dynamic_m': {
            'name': 'Dynamic m-axis test',
            'description': 'Verify the operator handles different m sizes at runtime.',
            'function': test_matmul_add_dynamic_m
        },
        'matmul_add::test_matmul_add_non_square': {
            'name': 'Non-square matrices test',
            'description': 'Verify correctness with non-square matrices and non-aligned dimensions.',
            'function': test_matmul_add_non_square
        },
        'matmul_add::test_matmul_add_edge_cases': {
            'name': 'Edge cases (zeros, single row, large values)',
            'description': 'Verify correctness on boundary inputs.',
            'function': test_matmul_add_edge_cases
        },
    }

    if args.list:
        print("\n" + "=" * 60)
        print("Available Tests for matmul_add Operator")
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
    print("PyPTO matmul_add Operator Tests")
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
            print("All matmul_add tests passed!")
            print("=" * 60)

    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
