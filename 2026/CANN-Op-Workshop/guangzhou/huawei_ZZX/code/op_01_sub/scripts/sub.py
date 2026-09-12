import numpy as np

def impl(x, y):
    # 验算函数：中间过程用 float64 计算，最后一次性转回输入 dtype，减少舍入误差
    dtype = x.dtype
    return (x.astype(np.float64) - y.astype(np.float64)).astype(dtype)

if __name__ == "__main__":
    for dtype in (np.float16, np.float32):
        x = np.random.uniform(1, 10, (8, 2)).astype(dtype)
        y = np.random.uniform(1, 10, (8, 2)).astype(dtype)
        z = impl(x, y)
        ref = (x.astype(np.float64) - y.astype(np.float64)).astype(dtype)
        ok = np.allclose(z, ref, atol=1e-2)
        print("dtype={}, shape={}, output dtype={}, match={}".format(
            np.dtype(dtype).name, x.shape, z.dtype, ok))