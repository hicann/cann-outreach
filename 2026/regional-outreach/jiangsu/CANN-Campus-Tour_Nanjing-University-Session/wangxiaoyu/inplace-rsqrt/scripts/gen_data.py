# ============================================================================
# 测试数据生成脚本 — InplaceRsqrt
# ============================================================================

import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 4D ND shape: [N4, N3, N2, N1], dtype=float16
shape = (2, 32, 128, 64)
total_length = int(np.prod(shape))
dtype = np.float16

# 生成输入数据（正值，避免 sqrt(负数) = NaN）
# 使用 [0.01, 100.0] 范围确保数值稳定
x = np.random.uniform(0.01, 100.0, total_length).astype(dtype)

x.tofile("input/input_self.bin")

# 计算 golden
golden = compute_golden(x)
golden.tofile("output/golden.bin")

print(f"Generated test data:")
print(f"  shape:  {shape}")
print(f"  dtype:  {dtype}")
print(f"  total:  {total_length} elements = {total_length * 2} bytes")
print(f"  input/input_self.bin:  min={x.min():.4f}, max={x.max():.4f}")
print(f"  output/golden.bin:     min={golden.min():.6f}, max={golden.max():.6f}")
