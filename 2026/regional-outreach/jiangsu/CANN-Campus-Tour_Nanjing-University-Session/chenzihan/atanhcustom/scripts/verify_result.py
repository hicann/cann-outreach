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
# 结果验证脚本 - Atanh 算子（float16）
# ============================================================================

import numpy as np
import sys

dtype = np.float16
rtol = 1e-3
atol = 1e-3   # float16 atanh 精度适当放宽

def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)
    
    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False
    
    if np.allclose(output, golden, rtol=rtol, atol=atol):
        print(f"Verification PASSED! Shape: {output.shape}")
        max_diff = np.max(np.abs(output.astype(np.float32) - golden.astype(np.float32)))
        print(f"Max diff: {max_diff}")
        return True
    else:
        diff = np.abs(output.astype(np.float32) - golden.astype(np.float32))
        print(f"Verification FAILED!")
        print(f"Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
        mismatches = np.where(diff > atol + rtol * np.abs(golden.astype(np.float32)))[0]
        print(f"Mismatch count: {len(mismatches)} / {len(golden)}")
        if len(mismatches) > 0:
            idx = mismatches[0]
            print(f"  First mismatch at index {idx}: output={output[idx]}, golden={golden[idx]}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)
    
    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
