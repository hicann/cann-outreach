import numpy as np

def impl(x):
    # 验算函数：逐元素 relu，float64 中间计算，最后一次性转回输入 dtype
    dtype = x.dtype
    return np.maximum(0, x.astype(np.float64)).astype(dtype)

if __name__ == "__main__":
    for dtype in (np.float16, np.float32):
        x = np.random.uniform(-10, 10, (8, 2)).astype(dtype)
        y = impl(x)
        ref = np.maximum(0, x.astype(np.float64)).astype(dtype)
        ok = np.array_equal(y, ref)
        print("dtype={}, shape={}, output dtype={}, match={}".format(
            np.dtype(dtype).name, x.shape, y.dtype, ok))
