# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# relu_custom Golden 计算

import torch

def golden_relu(x: torch.Tensor) -> torch.Tensor:
    """ReLU 参考实现: y = max(0, x)"""
    return torch.relu(x)


def generate_test_cases():
    cases = [
        ("T1_Random_Small_1x1x1x128",      (1, 1, 1, 128),    "random"),
        ("T2_Random_Medium_1x4x32x64",      (1, 4, 32, 64),    "random"),
        ("T3_Random_Large_2x8x64x128",      (2, 8, 64, 128),   "random"),
        ("T4_All_Positive_1x4x32x64",       (1, 4, 32, 64),    "positive"),
        ("T5_All_Negative_1x4x32x64",       (1, 4, 32, 64),    "negative"),
        ("T6_Mixed_Zero_1x4x32x64",         (1, 4, 32, 64),    "mixed_zero"),
    ]
    return cases


def generate_data(name: str, shape, data_type: str, seed: int = 42):
    torch.manual_seed(seed)

    if data_type == "random":
        x = (torch.rand(shape, dtype=torch.float16) * 2.0 - 1.0) * 10.0
    elif data_type == "positive":
        x = torch.rand(shape, dtype=torch.float16) * 10.0
    elif data_type == "negative":
        x = -torch.rand(shape, dtype=torch.float16) * 10.0
    elif data_type == "mixed_zero":
        x = (torch.rand(shape, dtype=torch.float16) * 2.0 - 1.0) * 10.0
        flat = x.flatten()
        flat[::5] = 0.0
        x = flat.reshape(shape)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    y = golden_relu(x)
    return x, y
