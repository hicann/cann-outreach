# ============================================================================
# Golden 计算 - ReLU 算子（float16）
# ============================================================================
# 被 gen_data.py 和 test_torch.py 共同引用
# ============================================================================

import numpy as np


def compute_golden(x):
    """计算 ReLU 算子的参考输出。

    relu(x) = max(0, x)

    Args:
        x: numpy array 或 torch.Tensor（float16）

    Returns:
        与输入同类型的 relu(x)
    """
    return np.maximum(0, x)
