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
# 结果验证脚本 - Relu 算子
# ============================================================================

import numpy as np
import sys

dtype = np.float16
rtol = 1e-3
atol = 1e-4

SHAPES = ["1_1_1_128", "1_4_32_64", "8_16_32_32"]


def verify_result(output_path, golden_path, shape_name):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"[{shape_name}] Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    if np.allclose(output, golden, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(output.astype(np.float32) - golden.astype(np.float32)))
        print(f"[{shape_name}] PASSED (Max diff={max_diff:.6f})")
        return True
    else:
        diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))
        print(f"[{shape_name}] FAILED!")
        print(f"  Max diff: {np.max(diff):.6f}, Mean diff: {np.mean(diff):.6f}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden.astype(np.float32)))[0]
        print(f"  Mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0:
            print(f"  First 5 mismatches indices: {mismatches[:5]}")
        return False


def main():
    all_passed = True
    for name in SHAPES:
        output_path = f"output/output_{name}.bin"
        golden_path = f"output/golden_{name}.bin"
        ok = verify_result(output_path, golden_path, name)
        if not ok:
            all_passed = False

    total = len(SHAPES)
    print(f"\n{'=' * 50}")
    print(f"Relu verification summary")
    print(f"{'=' * 50}")
    for name in SHAPES:
        import os
        ok = os.path.exists(f"output/output_{name}.bin")
        print(f"  [{name}]: {'VERIFIED' if ok else 'MISSING OUTPUT'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
