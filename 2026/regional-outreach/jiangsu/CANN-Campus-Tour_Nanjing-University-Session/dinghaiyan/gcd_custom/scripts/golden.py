# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================================
# Golden 计算 — Gcd 算子 (双通路共用)
# ============================================================================
#
# gcd(a, b) 的 float16 参考实现，使用 numpy 欧几里得逐元素计算。
#
# 与 kernel 对齐：
#   - float32 内部计算（避免 float16 溢出）
#   - MAX_GOLDEN_ITER = 24（与 MAX_GCD_ITER 一致）
#   - 防除零阈值 1e-10
# ============================================================================

import numpy as np


def compute_golden(self_arr, other_arr):
    """计算 Gcd 算子的参考输出：out[i] = gcd(|self[i]|, |other[i]|)

    Args:
        self_arr:  numpy array (float16), 形状已 broadcast
        other_arr: numpy array (float16), 形状已 broadcast

    Returns:
        numpy array (float16), gcd 结果
    """
    a = np.abs(self_arr).astype(np.float64)   # 用 float64 获得最精确的参考值
    b = np.abs(other_arr).astype(np.float64)

    # 欧几里得算法逐元素（上限 24 次，与 kernel 一致）
    MAX_GOLDEN_ITER = 24
    EPSILON = 1e-10
    mask = (b > EPSILON)
    for _ in range(MAX_GOLDEN_ITER):
        if not np.any(mask):
            break
        # a_next = b (for mask=True), keep a (for mask=False)
        a_next = np.where(mask, b, a)
        # r = a - floor(a / b) * b  (only for mask=True)
        r = np.where(mask, a - np.floor(a / b) * b, b)
        b_next = np.where(mask, r, b)

        a, b = a_next, b_next
        # 逼近收敛时值可能小于 EPSILON 但仍未达到精确解，留一点余量
        mask = (b > EPSILON)

    return a.astype(np.float16)
