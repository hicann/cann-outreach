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
# Golden 计算（双通路共用）- Atanh 算子
# ============================================================================

import numpy as np


def compute_golden(x):
    """计算算子的参考输出：y = atanh(x) = 0.5 * ln((1+x)/(1-x))

    Args:
        x: numpy array (float16，值域 (-1, 1))

    Returns:
        与输入同类型的参考输出
    """
    # numpy 的 arctanh 直接计算 atanh
    return np.arctanh(x)
