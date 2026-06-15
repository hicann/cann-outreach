# ============================================================================
# 结果验证脚本 — InplaceRsqrt
# ============================================================================

import numpy as np
import sys

dtype = np.float16
# FP16 容差: sqrt 和 reciprocal 各损失约 1 ULP，合计 2~3 ULP
rtol = 1e-3
atol = 1e-4


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    if np.allclose(output, golden, rtol=rtol, atol=atol):
        print(f"Verification PASSED! Shape: {output.shape}")
        diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))
        print(f"Max diff: {np.max(diff):.6f}, Mean diff: {np.mean(diff):.6f}")
        return True
    else:
        output_f32 = output.astype(np.float32)
        golden_f32 = golden.astype(np.float32)
        diff = np.abs(output_f32 - golden_f32)
        print(f"Verification FAILED!")
        print(f"Max diff: {np.max(diff):.6f}, Mean diff: {np.mean(diff):.6f}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden_f32))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0:
            idx = mismatches[0]
            print(f"  First mismatch at [{idx}]: "
                  f"output={output_f32[idx]:.6f}, golden={golden_f32[idx]:.6f}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
