#!/usr/bin/python3
# coding=utf-8
import numpy as np


def gen_golden_data_simple():
    # atanh domain: roughly (-0.99, -0.001) U (0.001, 0.99)
    x = np.random.uniform(0.01, 0.9, [2, 2, 4, 128]).astype(np.float16)
    golden = np.arctanh(x.astype(np.float32)).astype(np.float16)
    x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
