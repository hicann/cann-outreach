# InplaceRsqrt Golden 计算（FP16）
#
# 功能: self = 1 / sqrt(self)
# 用于 NPU 结果对比的参考实现

import numpy as np


def compute_golden(x):
    """计算 rsqrt golden 值

    Args:
        x: np.ndarray, dtype=np.float16, 输入张量

    Returns:
        np.ndarray, dtype=np.float16, rsqrt(x) 结果
    """
    # 转换为 float32 计算避免 FP16 精度损失
    x_f32 = x.astype(np.float32)
    result_f32 = 1.0 / np.sqrt(x_f32)
    return result_f32.astype(np.float16)
