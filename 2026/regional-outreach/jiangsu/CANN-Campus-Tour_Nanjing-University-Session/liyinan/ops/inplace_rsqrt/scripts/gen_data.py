#!/usr/bin/python3
# coding=utf-8
import numpy as np


def gen_golden_data_simple():
    # positive inputs for rsqrt
    x = np.random.uniform(0.25, 100, [2, 2, 4, 128]).astype(np.float16)
    golden = (1.0 / np.sqrt(x.astype(np.float32))).astype(np.float16)
    x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
