#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TruncateMod Kernel UT 数据生成（K1-K10，3 dtype × 广播 × 边界）。

输出文件命名约定（compare_data.py 按 dtype 前缀分组、golden/output 按 case 名配对）：
  {dtype}_input_{case}_x1.bin / {dtype}_input_{case}_x2.bin
  {dtype}_golden_{case}.bin / {dtype}_output_{case}.bin
Golden 计算：x1 - trunc(x1/x2)*x2，默认 float64 高精度后转输出 dtype；
K8（商值溢出 FP32）按 FP32 语义计算 golden（与设备 FP32 全链结果一致）。
"""

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

DTYPES = {
    "float16": np.float16,
    "bfloat16": bfloat16,
    "float32": np.float32,
}


def impl(x1, x2):
    """默认 golden：float64 高精度计算后由调用方转换 dtype。"""
    x1_64 = x1.astype(np.float64)
    x2_64 = x2.astype(np.float64)
    return x1_64 - np.trunc(x1_64 / x2_64) * x2_64


def impl_fp32_semantics(x1, x2):
    """FP32 语义 golden（K8 用）：全部在 float32 下运算，复现设备 FP32 全链行为
    （含 Inf 溢出，如 trunc(1e50→Inf)、Inf*tiny=Inf、x1-Inf=-Inf）。"""
    q = np.float32(x1) / np.float32(x2)
    q_trunc = np.trunc(q.astype(np.float32)).astype(np.float32)
    return np.float32(x1) - np.float32(q_trunc) * np.float32(x2)


class Case:
    def __init__(self, case_id, dtype, x1_shape, x2_shape, seed, kind="normal", impl_fn=impl):
        self.case_id = case_id
        self.dtype = dtype
        self.x1_shape = x1_shape
        self.x2_shape = x2_shape
        self.seed = seed
        self.kind = kind
        self.impl_fn = impl_fn

    def gen(self):
        rng = np.random.default_rng(self.seed)
        np_dtype = DTYPES[self.dtype]
        if self.kind == "normal":
            x1 = rng.uniform(-10.0, 10.0, self.x1_shape).astype(np_dtype)
            # 保证 |x2| >= 0.5 且非 0（避免 normal 用例商溢出/除零）
            x2 = rng.uniform(0.5, 5.0, self.x2_shape) * rng.choice([-1.0, 1.0], self.x2_shape)
            x2 = x2.astype(np_dtype)
        elif self.kind == "zero_divisor":
            x1 = rng.uniform(-10.0, 10.0, self.x1_shape).astype(np_dtype)
            x2 = rng.uniform(-5.0, 5.0, self.x2_shape).astype(np_dtype)
            # 部分 x2=0（IEEE：±Inf/NaN 按 golden 一致）
            x2[np.abs(x2) < 1.0] = 0.0
            x2 = x2.astype(np_dtype)
        elif self.kind == "big_quotient":
            # x1 大值 / x2 极小值 → 商远超 int32 范围（K8：FP32 全链，无 int32 溢出）
            x1 = rng.uniform(1e30, 1e32, self.x1_shape).astype(np.float32)
            x2 = rng.uniform(-1e-20, 1e-20, self.x2_shape).astype(np.float32)
            x2 = np.where(x2 == 0, 1e-20, x2).astype(np.float32)
        else:
            raise ValueError(f"unknown kind {self.kind}")

        x1 = np.broadcast_to(x1, self.x1_shape).astype(np_dtype)
        x2 = np.broadcast_to(x2, self.x2_shape).astype(np_dtype)
        out_shape = np.broadcast_shapes(self.x1_shape, self.x2_shape)
        b1 = np.broadcast_to(x1, out_shape)
        b2 = np.broadcast_to(x2, out_shape)

        golden = self.impl_fn(b1, b2)
        # 转输出 dtype：NaN/Inf 保留（K7 零除数 / K8 溢出场景）
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            golden = golden.astype(np_dtype)

        # 文件命名
        prefix = f"{self.dtype}_{self.case_id}"
        x1.astype(np_dtype).tofile(f"{prefix}_input_x1.bin")
        x2.astype(np_dtype).tofile(f"{prefix}_input_x2.bin")
        golden.astype(np_dtype).tofile(f"{prefix}_golden.bin")
        print(f"  {self.case_id}: dtype={self.dtype} x1={self.x1_shape} x2={self.x2_shape} "
              f"out={out_shape} n={golden.size}")


def main():
    for f in glob.glob("*.bin"):
        os.remove(f)

    # K1-K10（与 docs/PLAN.md §3.3 对齐）
    cases = [
        # K1: FP16 ELEWISE 非 32B 对齐（回归既有骨架用例）
        Case("K1", "float16", (5,), (5,), seed=1),
        # K2: FP16 ELEWISE 多核（blockFactor=1024 → 8 核）
        Case("K2", "float16", (8192,), (8192,), seed=2),
        # K3: BF16 ELEWISE 多核
        Case("K3", "bfloat16", (8192,), (8192,), seed=3),
        # K4: FP32 ELEWISE 多核
        Case("K4", "float32", (8192,), (8192,), seed=4),
        # K5: FP32 行广播（rowCount=1 单行 tile：x1=[1,64] 行重复）
        Case("K5", "float32", (1, 64), (64, 64), seed=5),
        # K6: FP16 标量广播（x2 标量 → 策略 0 Duplicate 展开）
        Case("K6", "float16", (1024,), (1,), seed=6),
        # K7: FP32 正负混合 + x2=0 边界（IEEE）
        Case("K7", "float32", (2048,), (2048,), seed=7, kind="zero_divisor"),
        # K8: FP32 大商值（FP32 语义 golden）
        Case("K8", "float32", (1024,), (1024,), seed=8, kind="big_quotient", impl_fn=impl_fp32_semantics),
        # K9: 非对齐 tile + 尾块（总元素数非 32B 倍数）三 dtype 各一
        Case("K9_fp16", "float16", (1003,), (1003,), seed=9),
        Case("K9_bf16", "bfloat16", (1003,), (1003,), seed=10),
        Case("K9_fp32", "float32", (1003,), (1003,), seed=11),
        # K10: BF16 左侧前缀广播（rowCount=1 单行 tile）
        Case("K10", "bfloat16", (1, 1, 32), (4, 5, 32), seed=12),
    ]
    for c in cases:
        c.gen()
    print(f"生成完成: {len(cases)} 个用例")


if __name__ == "__main__":
    main()
