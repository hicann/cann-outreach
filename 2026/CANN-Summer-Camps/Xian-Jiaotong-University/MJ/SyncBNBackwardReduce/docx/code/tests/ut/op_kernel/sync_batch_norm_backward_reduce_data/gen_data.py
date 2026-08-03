#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SyncBatchNormBackwardReduce 算子测试数据生成脚本

生成全场景测试数据（3种dtype × 多种shape），包括：
  - 输入 bin 文件（sum_dy, sum_dy_dx_pad, mean, invert_std）
  - golden bin 文件（sum_dy_xmu, y）

Golden 计算公式（与 kernel 对齐）：
  dy_mean      = mean * sum_dy            (Mul)
  sum_dy_xmu   = sum_dy_dx_pad - dy_mean  (Sub)
  y            = sum_dy_xmu * invert_std  (Mul)

对于 float16/bfloat16，先 cast 到 float32 计算，再 cast 回原 dtype（与 kernel CAST_RINT 对齐）。
"""

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

# dtype 映射表
D_TYPE_DICT = {
    "float32": np.float32,
    "float16": np.float16,
    "bfloat16": bfloat16,
    "float64": np.float64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "bool": np.bool_,
    "fp8_e4m3fn": np.uint8,
    "fp8_e5m2": np.uint8,
}

# ============================================================
# 测试用例矩阵: (case_id, dtype, shape, 描述)
# 覆盖: 3种dtype × 单核/多核 × 1D/多维 × 小/大shape
# ============================================================
TEST_CASES = [
    # float16 — 单核 (totalNum < 1024)
    ("c00", "float16",  (5,),       "fp16单核小shape"),
    ("c01", "float16",  (100,),     "fp16单核中shape"),
    ("c02", "float16",  (500,),     "fp16单核大shape(接近阈值)"),
    # float16 — 多核 (totalNum >= 1024)
    ("c03", "float16",  (1024,),    "fp16多核(阈值边界)"),
    ("c04", "float16",  (4096,),    "fp16多核中shape"),
    ("c05", "float16",  (65536,),   "fp16多核大shape"),
    # float32
    ("c06", "float32",  (5,),       "fp32单核小shape"),
    ("c07", "float32",  (4096,),    "fp32多核中shape"),
    ("c08", "float32",  (65536,),   "fp32多核大shape"),
    # bfloat16
    ("c09", "bfloat16", (5,),       "bf16单核小shape"),
    ("c10", "bfloat16", (4096,),    "bf16多核中shape"),
    ("c11", "bfloat16", (65536,),   "bf16多核大shape"),
    # 多维shape (展平后同1D处理)
    ("c12", "float16",  (2, 3),     "fp16多维shape单核"),
    ("c13", "float32",  (4, 1024),  "fp32多维shape多核"),
]


def generate_input_data(shape, dtype_str, seed_suffix):
    """
    生成有区分度的输入数据，使用固定种子确保可复现。
    生成 [-2.0, 2.0] 范围的浮点数据，确保计算结果有正负值。
    """
    np_type = D_TYPE_DICT[dtype_str]
    # 使用固定种子确保可复现
    seed = abs(hash(seed_suffix)) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    data = rng.uniform(-2.0, 2.0, size=shape).astype(np.float32)
    return data.astype(np_type)


def compute_golden(sum_dy, sum_dy_dx_pad, mean, invert_std, dtype_str):
    """
    计算 golden 数据，与 kernel 算法对齐。

    对于 float16/bfloat16: 先 cast 到 float32 计算，再 cast 回原 dtype
    对于 float32: 直接计算

    算法:
      dy_mean      = mean * sum_dy
      sum_dy_xmu   = sum_dy_dx_pad - dy_mean
      y            = sum_dy_xmu * invert_std
    """
    # 统一 cast 到 float32 进行计算
    sum_dy_f32 = sum_dy.astype(np.float32)
    sum_dy_dx_pad_f32 = sum_dy_dx_pad.astype(np.float32)
    mean_f32 = mean.astype(np.float32)
    invert_std_f32 = invert_std.astype(np.float32)

    # float32 计算
    dy_mean = mean_f32 * sum_dy_f32
    sum_dy_xmu_f32 = sum_dy_dx_pad_f32 - dy_mean
    y_f32 = sum_dy_xmu_f32 * invert_std_f32

    # cast 回原 dtype
    np_type = D_TYPE_DICT[dtype_str]
    sum_dy_xmu = sum_dy_xmu_f32.astype(np_type)
    y = y_f32.astype(np_type)

    return sum_dy_xmu, y


def save_bin(data, filename):
    """保存 numpy 数组到 bin 文件"""
    data.tofile(filename)


if __name__ == "__main__":
    # 清理旧 bin 文件
    for f in glob.glob("*.bin"):
        os.remove(f)

    print("=" * 60)
    print("SyncBatchNormBackwardReduce 测试数据生成")
    print("=" * 60)

    total_cases = 0
    for case_id, dtype_str, shape, desc in TEST_CASES:
        np_type = D_TYPE_DICT[dtype_str]

        # 生成输入数据（使用随机数据，固定种子确保可复现）
        sum_dy = generate_input_data(shape, dtype_str, case_id + "_sum_dy")
        sum_dy_dx_pad = generate_input_data(shape, dtype_str, case_id + "_pad")
        mean = generate_input_data(shape, dtype_str, case_id + "_mean")
        # invert_std 取绝对值 + 0.1 确保正值，避免除零/负标准差
        invert_std_raw = generate_input_data(shape, dtype_str, case_id + "_std")
        invert_std = np.abs(invert_std_raw.astype(np.float32)) + 0.1
        invert_std = invert_std.astype(np_type)

        # 计算 golden 数据
        golden_sum_dy_xmu, golden_y = compute_golden(
            sum_dy, sum_dy_dx_pad, mean, invert_std, dtype_str)

        # 保存输入数据
        save_bin(sum_dy, f"{dtype_str}_{case_id}_input_sync_batch_norm_backward_reduce_sum_dy.bin")
        save_bin(sum_dy_dx_pad, f"{dtype_str}_{case_id}_input_sync_batch_norm_backward_reduce_sum_dy_dx_pad.bin")
        save_bin(mean, f"{dtype_str}_{case_id}_input_sync_batch_norm_backward_reduce_mean.bin")
        save_bin(invert_std, f"{dtype_str}_{case_id}_input_sync_batch_norm_backward_reduce_invert_std.bin")

        # 保存 golden 数据
        save_bin(golden_sum_dy_xmu, f"{dtype_str}_{case_id}_golden_sync_batch_norm_backward_reduce_0.bin")
        save_bin(golden_y, f"{dtype_str}_{case_id}_golden_sync_batch_norm_backward_reduce_1.bin")

        total_cases += 1
        print(f"  [{case_id}] dtype={dtype_str:8s} shape={str(shape):12s} ({desc})")

    print("=" * 60)
    print(f"生成完成: 共 {total_cases} 个用例, {total_cases * 6} 个 bin 文件")
    print("=" * 60)
