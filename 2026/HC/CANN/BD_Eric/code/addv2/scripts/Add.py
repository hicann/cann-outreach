#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add 固定用例：两个 [256, 256] float16 张量逐元素相加。"""
import json
from pathlib import Path

import numpy as np

SHAPE = (256, 256)
DTYPE = np.dtype("float16")


def impl(x1, x2):
    """固定规格 golden：float32 中间加法，最近偶舍入返回 float16。"""
    if x1.shape != SHAPE or x2.shape != SHAPE:
        raise ValueError("本题两个输入的 shape 均须为 [256, 256]")
    if x1.dtype != DTYPE or x2.dtype != DTYPE:
        raise ValueError("本题两个输入的 dtype 均须为 float16")
    return (x1.astype(np.float32) + x2.astype(np.float32)).astype(np.float16)


if __name__ == "__main__":
    config = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
    assert config["cpu_cases"] == []
    assert len(config["npu_cases"]) == 1, "题目必须仅有1个测试用例"
    case = config["npu_cases"][0]
    for desc in case["input"] + case["output"]:
        assert tuple(desc["shape"]) == SHAPE and np.dtype(desc["datatype"]) == DTYPE
    assert SHAPE[-1] * DTYPE.itemsize % 32 == 0
    assert int(np.prod(SHAPE)) * DTYPE.itemsize % 32 == 0

    inputs = []
    for index, desc in enumerate(case["input"]):
        rng = np.random.default_rng(303 + index * 1000)
        inputs.append(rng.uniform(*desc["range"], size=SHAPE).astype(DTYPE))
    y = impl(*inputs)
    reference = (inputs[0].astype(np.float64) + inputs[1].astype(np.float64)).astype(DTYPE)
    assert y.shape == SHAPE and y.dtype == DTYPE
    np.testing.assert_array_equal(y, reference)
    print("Case 1 passed: x1/x2/y shape=[256, 256], dtype=float16, 65536 elements, 32-byte aligned lengths")
