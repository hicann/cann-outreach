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

# test_torch.py — PyTorch 通路测试

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from golden import generate_data, generate_test_cases

ATOL = 1e-3
RTOL = 1e-3


def main():
    print("==========================================")
    print("  relu_custom PyTorch 通路测试")
    print("  4D shape, float16, ND Format")
    print("==========================================")

    if not torch.npu.is_available():
        print("  SKIP: NPU 不可用")
        return 1

    so_path = os.path.join(os.path.dirname(__file__), "..", "build", "librelu_custom_ops.so")
    if not os.path.exists(so_path):
        print(f"ERROR: .so 文件不存在: {so_path}")
        return 1

    torch.ops.load_library(so_path)
    relu_custom = torch.ops.npu.relu_custom
    print(f"已加载: {so_path}")

    pass_count = 0
    total_count = 0

    for name, shape, data_type in generate_test_cases():
        x, golden_y = generate_data(name, shape, data_type)
        x_npu = x.npu()
        golden_y_npu = golden_y.npu()

        y_npu = relu_custom(x_npu)

        diff = torch.abs(golden_y_npu.float() - y_npu.float())
        max_diff = diff.max().item()
        threshold = ATOL + RTOL * torch.max(
            torch.abs(golden_y_npu.float()),
            torch.abs(y_npu.float()))
        errors = (diff > threshold).sum().item()

        status = "PASS" if errors == 0 else "FAIL"
        if errors == 0:
            pass_count += 1
        total_count += 1
        print(f"  {name}: {status} | MaxDiff={max_diff:.6e}")

    print(f"\n结果: {pass_count}/{total_count} PASS")
    return 0 if pass_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
