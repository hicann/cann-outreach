import numpy as np
import os

def gen_golden_data():
    # 构造 [N4, N3, N2, N1] 4维形状测试 (以2,3,32,32为例)
    shape = (2, 3, 32, 32)
    
    # ND Format, 满足 float16 要求
    input_x = np.random.uniform(-10.0, 10.0, size=shape).astype(np.float16)
    
    # CPU/Numpy 侧参考实现，生成 Golden 验证数据
    golden_y = np.maximum(input_x, 0.0).astype(np.float16)
    
    input_x.tofile("./input_x.bin")
    golden_y.tofile("./golden_y.bin")
    print(f"[成功] 已生成 float16 测试数据，Shape为: {shape}")

if __name__ == "__main__":
    gen_golden_data()
