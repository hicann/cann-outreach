# ============================================================================
# 测试数据生成 — Sign 算子
# ============================================================================

import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 4D ND shape: [N4, N3, N2, N1]
shape = (4, 64, 256, 32)
total_length = int(np.prod(shape))
dtype = np.float16

# 生成含正/负/零的混合数据
np.random.seed(42)
x = np.random.uniform(-10.0, 10.0, total_length).astype(dtype)

# 手动插入一些零值验证零值处理
x[::7] = 0.0  # 每 7 个元素设零

x.tofile("input/input_x.bin")

golden = compute_golden(x)
golden.tofile("output/golden.bin")

unique_vals = set(np.unique(golden))
print(f"Generated test data:")
print(f"  shape:     {shape}")
print(f"  dtype:     {dtype}")
print(f"  total:     {total_length}")
print(f"  input:     min={x.min():.4f}, max={x.max():.4f}")
print(f"  golden:    unique values={sorted(unique_vals)}")
pos = np.sum(golden == 1.0).item()
neg = np.sum(golden == -1.0).item()
zero = np.sum(golden == 0.0).item()
print(f"  sign dist: +1:{pos} ({100*pos/total_length:.1f}%), "
      f"-1:{neg} ({100*neg/total_length:.1f}%), "
      f" 0:{zero} ({100*zero/total_length:.1f}%)")
