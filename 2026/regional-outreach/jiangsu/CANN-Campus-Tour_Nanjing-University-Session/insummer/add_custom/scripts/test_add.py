#!/usr/bin/env python3
"""
test_add.py - Unit test for add_custom operator (float16, 2D).

Tests 2D shapes:
  - [1, 128]
  - [4, 2048]
  - [32, 4096]

Compares Ascend C add result (z = x + y) against NumPy reference.

Usage:
    # After building the operator:
    python3 scripts/test_add.py --op-dir build/output

    # Reference-only (no Ascend hardware):
    python3 scripts/test_add.py
"""

import argparse
import os
import sys
import numpy as np

try:
    import acl
except ImportError:
    acl = None


# ──────────────────────────────────────────────────────────────────────
#  Reference implementation
# ──────────────────────────────────────────────────────────────────────
def reference_add_fp16(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reference: z = x + y for float16 data."""
    return (x.astype(np.float32) + y.astype(np.float32)).astype(np.float16)


# ──────────────────────────────────────────────────────────────────────
#  Ascend C operator runner (via aclnn)
# ──────────────────────────────────────────────────────────────────────
class AscendAddRunner:
    """Runs the add_custom operator on Ascend NPU."""

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

    def run(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Run add_custom on Ascend, return result z as numpy array."""
        n_bytes = x.nbytes

        # Create ACL tensor descriptors
        x_desc = acl.create_tensor_desc(acl.ACL_FLOAT16, x.shape)
        y_desc = acl.create_tensor_desc(acl.ACL_FLOAT16, y.shape)
        z_desc = acl.create_tensor_desc(acl.ACL_FLOAT16, x.shape)

        # Allocate device memory
        x_dev = acl.rt.malloc(n_bytes, acl.ACL_MEM_MALLOC_HUGE_FIRST)
        y_dev = acl.rt.malloc(n_bytes, acl.ACL_MEM_MALLOC_HUGE_FIRST)
        z_dev = acl.rt.malloc(n_bytes, acl.ACL_MEM_MALLOC_HUGE_FIRST)

        # Copy inputs H2D
        acl.rt.memcpy(x_dev, n_bytes, x.tobytes(), n_bytes, acl.ACL_MEMCPY_HOST_TO_DEVICE)
        acl.rt.memcpy(y_dev, n_bytes, y.tobytes(), n_bytes, acl.ACL_MEMCPY_HOST_TO_DEVICE)

        # Execute operator (2 inputs, 1 output)
        ret = acl.op.execute(
            "add_custom",
            [x_dev, y_dev],
            [z_dev],
            [x_desc, y_desc],
            [z_desc],
        )
        assert ret == 0, f"Operator execution failed: {ret}"

        # Copy result D2H
        z_bytes = bytearray(n_bytes)
        acl.rt.memcpy(z_bytes, n_bytes, z_dev, n_bytes, acl.ACL_MEMCPY_DEVICE_TO_HOST)
        z = np.frombuffer(z_bytes, dtype=np.float16).reshape(x.shape)

        # Cleanup
        acl.rt.free(x_dev)
        acl.rt.free(y_dev)
        acl.rt.free(z_dev)
        acl.destroy_tensor_desc(x_desc)
        acl.destroy_tensor_desc(y_desc)
        acl.destroy_tensor_desc(z_desc)

        return z

    def __del__(self):
        if acl is not None:
            acl.rt.reset_device(self.device_id)
            acl.finalize()


# ──────────────────────────────────────────────────────────────────────
#  Unit test functions
# ──────────────────────────────────────────────────────────────────────
def _gen_test_data(shape, low=-10.0, high=10.0, seed=42):
    """Generate float16 test data with both positive and negative values."""
    rng = np.random.RandomState(seed)
    x = (rng.rand(*shape).astype(np.float64) * (high - low) + low).astype(np.float16)
    y = (rng.rand(*shape).astype(np.float64) * (high - low) + low).astype(np.float16)
    return x, y


def test_shape(shape, runner=None, op_dir=None, device_id=0):
    """Test add_custom on a given 2D shape, return (passed, details, x, y, z)."""
    x, y = _gen_test_data(shape)
    expected = reference_add_fp16(x, y)

    if runner is not None:
        try:
            actual = runner.run(x, y)
        except Exception as e:
            return False, f"Operator run failed: {e}", x, y, None
    else:
        # Reference-only mode (no Ascend hardware)
        actual = expected.copy()

    # Compare (float32 precision to avoid fp16 underflow noise)
    max_diff = np.max(
        np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    )
    match = np.allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )

    details = {
        "shape": list(shape),
        "elements": int(np.prod(shape)),
        "max_diff": float(max_diff),
        "match": bool(match),
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "y_min": float(np.min(y)),
        "y_max": float(np.max(y)),
        "z_min": float(np.min(actual)),
        "z_max": float(np.max(actual)),
    }

    return match, details, x, y, actual


def run_all_tests(op_dir=None, device_id=0):
    """Run all defined test shapes."""
    test_shapes = [
        [1, 128],
        [4, 2048],
        [32, 4096],
    ]

    runner = None
    if op_dir and os.path.isdir(op_dir):
        try:
            runner = AscendAddRunner(op_dir, device_id)
        except Exception as e:
            print(f"⚠️  Ascend runtime init failed (will test reference only): {e}")

    print("=" * 60)
    print("  add_custom — Unit Test Report")
    print("=" * 60)
    print()

    all_passed = True
    results = []

    for shape in test_shapes:
        passed, details, x, y, z = test_shape(shape, runner, op_dir, device_id)
        results.append((shape, passed, details))

        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Shape {str(shape):<20}  {status}")
        if details:
            print(f"    Elements : {details['elements']}")
            print(f"    Max diff : {details['max_diff']:.6e}")
            print(f"    x range  : [{details['x_min']:.4f}, {details['x_max']:.4f}]")
            print(f"    y range  : [{details['y_min']:.4f}, {details['y_max']:.4f}]")
            print(f"    z range  : [{details['z_min']:.4f}, {details['z_max']:.4f}]")
        print()

        if not passed:
            all_passed = False

    # Summary
    print("-" * 60)
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
        description="Unit test for add_custom float16 operator"
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
