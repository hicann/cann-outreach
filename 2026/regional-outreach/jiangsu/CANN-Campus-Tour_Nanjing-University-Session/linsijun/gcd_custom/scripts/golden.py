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
# Golden 计算 - Gcd Float16
# ============================================================================
#
# numpy 实现逐元素欧几里得算法求最大公约数。
# 支持 float16 精度，对整数型浮点输入给出正确结果，
# 对非整数浮点输入返回浮点欧几里得算法的近似结果。
# ============================================================================

import numpy as np


def compute_golden(self_data, other_data):
    """计算 gcd 算子的参考输出: 逐元素欧几里得算法

    Args:
        self_data:  numpy array (float16) 或 torch.Tensor (已 broadcast)
        other_data: numpy array (float16) 或 torch.Tensor (已 broadcast)

    Returns:
        与输入同类型的参考输出: gcd(|self|, |other|)
    """
    # 转为 numpy 计算
    if hasattr(self_data, 'numpy') or hasattr(self_data, 'cpu'):
        # 可能是 torch.Tensor
        import torch
        if isinstance(self_data, torch.Tensor):
            self_data = self_data.cpu().numpy()
            other_data = other_data.cpu().numpy()

    a = np.abs(self_data.astype(np.float32))
    b = np.abs(other_data.astype(np.float32))

    # 欧几里得算法: while b != 0: a, b = b, a mod b
    max_iter = 64
    for _ in range(max_iter):
        # 处理 b 接近 0 的情况
        b_nonzero = np.abs(b) > 1e-8
        
        # 仅对 b != 0 的元素计算 a mod b
        quotient = np.zeros_like(a)
        remainder = a.copy()
        
        # floor(a / b) 对非零 b 计算
        quotient[b_nonzero] = np.floor(a[b_nonzero] / b[b_nonzero])
        remainder[b_nonzero] = a[b_nonzero] - b[b_nonzero] * quotient[b_nonzero]
        
        # 对于 b == 0 的元素, remainder 保持为 a (gcd 已在 a 中)
        
        # 交换并确保 a >= b
        a_new = np.maximum(b, remainder)
        b_new = np.minimum(b, remainder)
        a, b = a_new, b_new

    result = a
    return result.astype(np.float16)
