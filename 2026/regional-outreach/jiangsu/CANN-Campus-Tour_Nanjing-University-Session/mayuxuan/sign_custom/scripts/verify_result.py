# ============================================================================
# 结果验证 — Sign 算子
# ============================================================================

import numpy as np
import sys

dtype = np.float16

# Sign 输出是精确的 ±1.0 / 0.0，容差设极严
rtol = 0.0
atol = 1e-6


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    success = np.allclose(output, golden, rtol=rtol, atol=atol)
    diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))

    if success:
        print(f"Verification PASSED! Shape: {output.shape}")
        print(f"Max diff: {np.max(diff):.8f}")
        return True
    else:
        print(f"Verification FAILED!")
        print(f"Max diff: {np.max(diff):.8f}, Mean diff: {np.mean(diff):.8f}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0:
            idx = mismatches[:5]
            for i in idx:
                print(f"  [{i}]: output={output[i]}, golden={golden[i]}, diff={diff[i]:.8f}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
