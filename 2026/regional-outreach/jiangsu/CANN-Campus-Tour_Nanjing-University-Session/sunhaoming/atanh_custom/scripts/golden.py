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
# Golden 计算 - atanh 算子
# ============================================================================
#
# atanh(x) = 0.5 * ln((1 + x) / (1 - x)), x ∈ (-1, 1)
#
# 输入: float32（精度对齐 half 范围）
# 输出: float32，再由 gen_data.py 转为 float16 保存
# ============================================================================

import numpy as np


def compute_golden(x):
    """计算 atanh 算子的参考输出。

    Args:
        x: numpy array (float32 类型，范围与 float16 一致)

    Returns:
        atanh(x) 的 numpy array
    """
    return np.arctanh(x)
