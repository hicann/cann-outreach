# ============================================================================
# 测试数据生成脚本 - ReLU 算子（float16, 4D ND）
# ============================================================================
#
# relu(x) = max(0, x)，对所有实数都有定义
# 生成正负混合数据，覆盖激活/未激活两种状态
#
# 4D Shape: 通过环境变量传入（默认 [8,16,128,256]）
#   SHAPE_N1=16 SHAPE_N2=32 SHAPE_N3=64 SHAPE_N4=128 python3 gen_data.py
# ============================================================================

import numpy as np
import os

from golden import compute_golden

os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 4D Shape（支持环境变量覆盖）
N1 = int(os.environ.get("SHAPE_N1", "8"))
N2 = int(os.environ.get("SHAPE_N2", "16"))
N3 = int(os.environ.get("SHAPE_N3", "128"))
N4 = int(os.environ.get("SHAPE_N4", "256"))
total_length = N1 * N2 * N3 * N4

dtype = np.float16

# 生成正负混合数据（约 50% 正, 50% 负）
np.random.seed(42)
x = (np.random.randn(total_length).astype(np.float32) * 3.0).astype(dtype)
x.tofile("input/input_x.bin")

# 计算 golden
golden = compute_golden(x)
golden.tofile("output/golden.bin")

print(f"Generated test data: shape=({N1},{N2},{N3},{N4}), total={total_length}, dtype={dtype}")
print(f"  input/input_x.bin: range=[{x.min():.4f}, {x.max():.4f}]")
print(f"    positive: {(x > 0).sum()}, negative: {(x < 0).sum()}, zero: {(x == 0).sum()}")
print(f"  output/golden.bin: range=[{golden.min():.4f}, {golden.max():.4f}]")
