#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# verify_result.py — 验证可执行文件输出

import os
import sys
import numpy as np

ATOL = 1e-3
RTOL = 1e-3

TEST_CASES = [
    "T1_Random_Small_1x1x1x128",
    "T2_Random_Medium_1x4x32x64",
    "T3_Random_Large_2x8x64x128",
    "T4_All_Positive_1x4x32x64",
    "T5_All_Negative_1x4x32x64",
    "T6_Mixed_Zero_1x4x32x64",
]


def verify(golden_path: str, actual_path: str, name: str = "") -> bool:
    golden = np.fromfile(golden_path, dtype=np.float16)
    actual = np.fromfile(actual_path, dtype=np.float16)

    if len(golden) != len(actual):
        print(f"  FAIL[{name}]: 长度不匹配: golden={len(golden)}, actual={len(actual)}")
        return False

    diff = np.abs(golden.astype(np.float32) - actual.astype(np.float32))
    max_diff = np.max(diff)
    threshold = ATOL + RTOL * np.maximum(
        np.abs(golden.astype(np.float32)),
        np.abs(actual.astype(np.float32)))
    errors = np.sum(diff > threshold)

    print(f"  {name}: MaxDiff={max_diff:.6e}, Errors={errors}/{len(golden)}")

    if errors > 0:
        for idx in np.where(diff > threshold)[0][:10]:
            print(f"    [{idx}]: golden={golden[idx]}, actual={actual[idx]}, diff={diff[idx]:.6e}")
        return False
    return True


def main():
    test_dir = os.path.join(os.path.dirname(__file__), "..", "test")
    actual_dir = os.path.join(os.path.dirname(__file__), "..", "build")

    all_pass = True
    for name in TEST_CASES:
        actual_path = os.path.join(actual_dir, f"{name}_output.bin")
        golden_path = os.path.join(test_dir, f"{name}_golden.bin")
        if not os.path.exists(actual_path):
            print(f"  {name}: 输出文件不存在")
            all_pass = False
            continue
        if not verify(golden_path, actual_path, name):
            all_pass = False

    print(f"\n{'全部通过！' if all_pass else '存在失败用例！'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
