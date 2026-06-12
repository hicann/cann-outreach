# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================================
# 结果验证脚本 — Gcd 算子
# ============================================================================
#
# float16 精度容差（欧几里得算法浮点累计误差）:
#   rtol = 1e-2     # 相对容差
#   atol = 1e-4     # 绝对容差
# ============================================================================

import numpy as np
import sys

# 验证参数
dtype = np.float16
rtol = 1e-2          # 相对容差 (GCD 浮点算法累计误差)
atol = 1e-4          # 绝对容差


def verify_result(output_path, golden_path):
    output = np.fromfile(output_path, dtype=dtype)
    golden = np.fromfile(golden_path, dtype=dtype)

    if output.shape != golden.shape:
        print(f"Shape mismatch: output {output.shape} vs golden {golden.shape}")
        return False

    output_f32 = output.astype(np.float32)
    golden_f32 = golden.astype(np.float32)

    abs_diff = np.abs(output_f32 - golden_f32)
    rel_diff = abs_diff / (np.abs(golden_f32) + 1e-10)

    max_abs = np.max(abs_diff)
    max_rel = np.max(rel_diff)

    # 逐元素检查
    failing = (abs_diff > atol) & (rel_diff > rtol)
    n_fail = np.sum(failing)
    n_total = golden.size

    if n_fail == 0:
        print(f"Verification PASSED! Shape: {output.shape}")
        print(f"  Max abs diff: {max_abs:.6e}")
        print(f"  Max rel diff: {max_rel:.6e}")
        return True
    else:
        print(f"Verification FAILED!")
        print(f"  Max abs diff: {max_abs:.6e}")
        print(f"  Max rel diff: {max_rel:.6e}")
        print(f"  Mismatch count: {n_fail} / {n_total}")

        # 打印前 5 个不匹配位置
        fail_indices = np.where(failing)[0][:5]
        for idx in fail_indices:
            print(f"  [{idx}] output={output[idx]}, golden={golden[idx]}, "
                  f"diff={abs_diff[idx]:.6e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python verify_result.py <output.bin> <golden.bin>")
        sys.exit(1)

    success = verify_result(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
