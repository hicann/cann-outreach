#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立校验 add_custom 算子的运行结果。

校验内容：
    1. 结果文件大小与 shape 匹配；
    2. output/output_z.bin 与 input/golden_z.bin 逐元素比对；
    3. 与 input_x.bin + input_y.bin 现场重算的真值比对。

用法：python3 scripts/verify_result.py [N2] [N1]
退出码：0 表示校验通过，1 表示失败
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(ROOT_DIR, "input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")

RELATIVE_ERROR_THRESHOLD = 1e-3
ABSOLUTE_ERROR_THRESHOLD = 1e-3
ERROR_RATIO_THRESHOLD = 0.0  # 允许的误差元素占比，0 表示要求全部元素达标


def compare(actual, expect, name):
    """返回 (是否通过, 最大相对误差, 误差元素个数)"""
    actual_f = actual.astype(np.float32)
    expect_f = expect.astype(np.float32)
    abs_error = np.abs(actual_f - expect_f)
    denom = np.where(np.abs(expect_f) > 1e-6, np.abs(expect_f), 1.0)
    relative_error = np.where(np.abs(expect_f) > 1e-6, abs_error / denom, abs_error)
    failed_mask = (relative_error > RELATIVE_ERROR_THRESHOLD) & (abs_error > ABSOLUTE_ERROR_THRESHOLD)
    error_count = int(np.count_nonzero(failed_mask))
    max_relative_error = float(np.max(relative_error)) if relative_error.size else 0.0

    print("[INFO] compare {}: total = {}, error count = {}, max relative error = {:.6e}".format(
        name, actual.size, error_count, max_relative_error))
    if error_count:
        index = np.flatnonzero(failed_mask.reshape(-1))[:5]
        for i in index:
            print("[ERROR]   index {}: actual = {}, expect = {}".format(
                i, actual_f.reshape(-1)[i], expect_f.reshape(-1)[i]))

    passed = error_count <= int(ERROR_RATIO_THRESHOLD * actual.size)
    return passed, max_relative_error, error_count


def main():
    n2 = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    total = n2 * n1

    output_path = os.path.join(OUTPUT_DIR, "output_z.bin")
    golden_path = os.path.join(INPUT_DIR, "golden_z.bin")
    x_path = os.path.join(INPUT_DIR, "input_x.bin")
    y_path = os.path.join(INPUT_DIR, "input_y.bin")

    for path in (output_path, golden_path, x_path, y_path):
        if not os.path.exists(path):
            print("[ERROR] file not found: {}".format(path))
            return 1

    def load(path):
        data = np.fromfile(path, dtype=np.float16)
        if data.size != total:
            raise ValueError("{} size mismatch: expect {} elements, actual {}".format(
                path, total, data.size))
        return data

    try:
        output = load(output_path)
        golden = load(golden_path)
        x = load(x_path)
        y = load(y_path)
    except ValueError as err:
        print("[ERROR] {}".format(err))
        return 1

    print("[INFO] shape = [{}, {}], dtype = float16, format = ND".format(n2, n1))
    ok_golden, max_err_golden, _ = compare(output, golden, "output_z vs golden_z")
    recomputed = (x.astype(np.float32) + y.astype(np.float32)).astype(np.float16)
    ok_recomputed, max_err_recomputed, _ = compare(output, recomputed, "output_z vs x + y (recomputed)")

    if ok_golden and ok_recomputed:
        print("[INFO] verification PASSED")
        return 0
    print("[ERROR] verification FAILED, max relative error = {:.6e}".format(
        max(max_err_golden, max_err_recomputed)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
