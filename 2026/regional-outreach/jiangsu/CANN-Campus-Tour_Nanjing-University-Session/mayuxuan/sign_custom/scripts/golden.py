# Sign 算子 Golden 计算 (FP16)
# sign(x) = 1.0 (x>0), 0.0 (x==0), -1.0 (x<0)

import numpy as np


def compute_golden(x):
    """计算 sign golden 值

    Args:
        x: np.ndarray, dtype=np.float16

    Returns:
        np.ndarray, dtype=np.float16, sign(x)
    """
    # 直接使用 numpy 的 sign 函数
    return np.sign(x).astype(np.float16)
