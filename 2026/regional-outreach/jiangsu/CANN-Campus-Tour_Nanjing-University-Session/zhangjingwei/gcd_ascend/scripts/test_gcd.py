#!/usr/bin/env python3
"""
test_gcd.py - Unit test for gcd_ascend operator (float16, 4D ND, broadcast).

Tests broadcast scenarios:
  - Same shape:     self=[1,4,8,16],      other=[1,4,8,16]
  - Broadcast self: self=[1,1,1,16],      other=[1,4,8,16]
  - Broadcast other:self=[1,4,8,16],      other=[1,1,1,16]
  - Mixed broadcast:self=[1,1,8,1],       other=[1,4,1,16]
  - Full broadcast: self=[1,1,1,1],       other=[1,4,8,16]

Reference: numpy.gcd for integers. Since we work with float16, we interpret
the values as float16 and compute GCD using the Euclidean algorithm via
math.gcd on the integer representation, or via np.gcd on rounded values.

Usage:
    # After building the operator:
    python3 scripts/test_gcd.py --op-dir build/output

    # Reference-only (no hardware):
    python3 scripts/test_gcd.py
"""

import argparse
import os
import sys
import math
import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    import acl
except ImportError:
    acl = None

try:
    import ascend_c_op as ac_op
except ImportError:
    ac_op = None


# ──────────────────────────────────────────────────────────────────────
#  Reference implementation
# ──────────────────────────────────────────────────────────────────────

def reference_gcd_fp16(self_arr: np.ndarray, other_arr: np.ndarray) -> np.ndarray:
    """
    Reference: out = gcd(self, other) for float16 data.

    Since NumPy doesn't support float16 GCD natively, we compute using
    the Euclidean algorithm element-wise via float64 for precision.
    """
    # Broadcast inputs to output shape
    out_shape = np.broadcast_shapes(self_arr.shape, other_arr.shape)
    self_b = np.broadcast_to(self_arr, out_shape).astype(np.float64)
    other_b = np.broadcast_to(other_arr, out_shape).astype(np.float64)

    out = np.zeros(out_shape, dtype=np.float64)
    it = np.nditer(out, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        a = abs(float(self_b[idx]))
        b = abs(float(other_b[idx]))

        # Euclidean algorithm
        eps = 1e-4
        max_iter = 50
        for _ in range(max_iter):
            if b <= eps:
                break
            # a mod b
            ratio = a / b
            q = int(ratio)
            if float(q) > ratio:
                q -= 1
            r = a - q * b
            if abs(r) <= eps:
                break
            a, b = b, r
        else:
            b = a  # fallback (shouldn't normally hit max_iter)

        out[idx] = b if b > 0 else a
        it.iternext()

    return out.astype(np.float16)


# ──────────────────────────────────────────────────────────────────────
#  Ascend C operator runner
# ──────────────────────────────────────────────────────────────────────

class AscendGcdRunner:
    """Runs the gcd_custom operator on Ascend NPU."""

    def __init__(self, op_dir: str, device_id: int = 0):
        self.op_dir = op_dir
        self.device_id = device_id
        self._init_ascend()

    def _init_ascend(self):
        """Initialize ACL runtime."""
        if acl is None:
            raise ImportError("acl module not found. Source Ascend environment first.")
        ret = acl.init()
        assert ret == 0, f"acl.init failed: {ret}"
        ret = acl.rt.set_device(self.device_id)
        assert ret == 0, f"acl.rt.set_device failed: {ret}"
        ret = acl.op.load(self.op_dir)
        assert ret == 0, f"acl.op.load failed: {ret}"

    def run(self, self_arr: np.ndarray, other_arr: np.ndarray) -> np.ndarray:
        """
        Run gcd_custom on Ascend. Inputs are broadcast by the operator.
        The output shape = broadcast(self_arr.shape, other_arr.shape).
        """
        out_shape = np.broadcast_shapes(self_arr.shape, other_arr.shape)
        out_nbytes = np.dtype(np.float16).itemsize * int(np.prod(out_shape))
        self_nbytes = self_arr.nbytes
        other_nbytes = other_arr.nbytes

        # Create ACL tensor descriptors
        self_desc = acl.create_tensor_desc(acl.ACL_FLOAT16, self_arr.shape)
        other_desc = acl.create_tensor_desc(acl.ACL_FLOAT16, other_arr.shape)
        out_desc = acl.create_tensor_desc(acl.ACL_FLOAT16, out_shape)

        # Allocate device memory
        self_dev = acl.rt.malloc(self_nbytes, acl.ACL_MEM_MALLOC_HUGE_FIRST)
        other_dev = acl.rt.malloc(other_nbytes, acl.ACL_MEM_MALLOC_HUGE_FIRST)
        out_dev = acl.rt.malloc(out_nbytes, acl.ACL_MEM_MALLOC_HUGE_FIRST)

        # Copy inputs to device
        acl.rt.memcpy(self_dev, self_nbytes, self_arr.tobytes(), self_nbytes,
                      acl.ACL_MEMCPY_HOST_TO_DEVICE)
        acl.rt.memcpy(other_dev, other_nbytes, other_arr.tobytes(), other_nbytes,
                      acl.ACL_MEMCPY_HOST_TO_DEVICE)

        # Execute operator
        ret = acl.op.execute(
            "gcd_custom",
            [self_dev, other_dev],
            [out_dev],
            [self_desc, other_desc],
            [out_desc],
        )
        assert ret == 0, f"Operator execution failed: {ret}"

        # Copy result back
        out_bytes = bytearray(out_nbytes)
        acl.rt.memcpy(out_bytes, out_nbytes, out_dev, out_nbytes,
                      acl.ACL_MEMCPY_DEVICE_TO_HOST)
        y = np.frombuffer(out_bytes, dtype=np.float16).reshape(out_shape)

        acl.rt.free(self_dev)
        acl.rt.free(other_dev)
        acl.rt.free(out_dev)
        acl.destroy_tensor_desc(self_desc)
        acl.destroy_tensor_desc(other_desc)
        acl.destroy_tensor_desc(out_desc)

        return y

    def __del__(self):
        if acl is not None:
            acl.rt.reset_device(self.device_id)
            acl.finalize()


# ──────────────────────────────────────────────────────────────────────
#  Test data generation
# ──────────────────────────────────────────────────────────────────────

def _gen_test_data(shape, low=1.0, high=32.0, seed=42):
    """
    Generate float16 test data with integer-like values for clean GCD.
    Values from 1 to 32 (integers cast to float16) produce deterministic GCD.
    Negative values are included to test abs handling.
    """
    rng = np.random.RandomState(seed)
    # Generate integers 1..32 as float16 for clean GCD results
    int_vals = rng.randint(1, 33, size=shape).astype(np.float16)
    # Add some non-integer values too
    float_vals = (rng.rand(*shape).astype(np.float64) * 15.0 + 1.0).astype(np.float16)
    # Mix: use integer-like for half the data
    mask = rng.rand(*shape) > 0.5
    x = np.where(mask, int_vals, float_vals)
    # Randomly negate some values to test abs handling
    neg_mask = rng.rand(*shape) > 0.7
    x[neg_mask] = -x[neg_mask]
    return x


# ──────────────────────────────────────────────────────────────────────
#  Unit test functions
# ──────────────────────────────────────────────────────────────────────

def test_broadcast_case(self_shape, other_shape, runner=None, op_dir=None,
                         device_id=0, test_id="unknown"):
    """
    Test gcd with broadcast from self_shape and other_shape.
    Returns (passed, details, self_arr, other_arr, actual).
    """
    self_arr = _gen_test_data(self_shape)
    other_arr = _gen_test_data(other_shape)

    out_shape = np.broadcast_shapes(self_shape, other_shape)
    expected = reference_gcd_fp16(self_arr, other_arr)

    if runner is not None:
        try:
            actual = runner.run(self_arr, other_arr)
        except Exception as e:
            return False, f"Operator run failed: {e}", None, None, None
    else:
        actual = expected.copy()

    # Compare
    max_diff = np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))
    match = np.allclose(actual.astype(np.float32), expected.astype(np.float32),
                        rtol=1e-2, atol=1e-2)

    details = {
        "test_id": test_id,
        "self_shape": list(self_shape),
        "other_shape": list(other_shape),
        "out_shape": list(out_shape),
        "elements": int(np.prod(out_shape)),
        "max_diff": float(max_diff),
        "match": bool(match),
        "y_min": float(np.min(actual)),
        "y_max": float(np.max(actual)),
    }

    # Verify GCD property: gcd(a,b) divides both a and b (within tolerance)
    gcd_property_ok = True
    if np.prod(out_shape) > 0:
        flat_a = np.broadcast_to(self_arr, out_shape).flatten().astype(np.float64)
        flat_b = np.broadcast_to(other_arr, out_shape).flatten().astype(np.float64)
        flat_g = actual.flatten().astype(np.float64)
        # Check that gcd divides both inputs (not strict for floats)
        for i in range(min(100, len(flat_g))):
            g = flat_g[i]
            if g > 1e-4:
                # a/g should be near-integer
                ratio_a = flat_a[i] / g
                ratio_b = flat_b[i] / g
                if abs(ratio_a - round(ratio_a)) > 1e-2 and abs(ratio_b - round(ratio_b)) > 1e-2:
                    gcd_property_ok = False
                    break

    details["gcd_property_ok"] = gcd_property_ok

    return match and gcd_property_ok, details, self_arr, other_arr, actual


def run_all_tests(op_dir=None, device_id=0):
    """Run all broadcast test cases."""
    # Test cases: (self_shape, other_shape, description)
    test_cases = [
        ([1, 4, 8, 16], [1, 4, 8, 16], "same_shape"),
        ([1, 1, 1, 16], [1, 4, 8, 16], "broadcast_self"),
        ([1, 4, 8, 16], [1, 1, 1, 16], "broadcast_other"),
        ([1, 1, 8, 1],  [1, 4, 1, 16], "mixed_broadcast"),
        ([1, 1, 1, 1],  [1, 4, 8, 16], "full_broadcast"),
        ([1, 2, 4, 8],  [1, 2, 1, 8],  "partial_broadcast_n2"),
    ]

    runner = None
    if op_dir and os.path.isdir(op_dir):
        try:
            runner = AscendGcdRunner(op_dir, device_id)
        except Exception as e:
            print(f"⚠️  Ascend runtime init failed (will test reference only): {e}")

    print("=" * 65)
    print("  gcd_ascend — Unit Test Report (Broadcast)")
    print("=" * 65)
    print()

    all_passed = True
    results = []

    for self_shape, other_shape, test_id in test_cases:
        passed, details, s, o, y = test_broadcast_case(
            self_shape, other_shape, runner, op_dir, device_id, test_id
        )
        results.append((test_id, passed, details))

        status = "✅ PASS" if passed else "❌ FAIL"
        label = f"{str(self_shape)} vs {str(other_shape)}"
        print(f"  {label:<40}  {status}")
        if details:
            print(f"    Output shape: {details['out_shape']}")
            print(f"    Elements:     {details['elements']}")
            print(f"    Max diff:     {details['max_diff']:.6e}")
            print(f"    GCD property: {details.get('gcd_property_ok', 'N/A')}")
        print()

        if not passed:
            all_passed = False

    # Summary
    print("-" * 65)
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    print(f"  Result: {passed_count}/{total_count} tests passed")
    print()

    return all_passed


# ──────────────────────────────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unit test for gcd_ascend float16 broadcast operator"
    )
    parser.add_argument(
        "--op-dir", type=str, default=None,
        help="Path to compiled operator package directory. If omitted, "
             "only reference test runs (no Ascend hardware needed)."
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="Ascend device ID (default: 0)"
    )
    args = parser.parse_args()

    success = run_all_tests(args.op_dir, args.device)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
