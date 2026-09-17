#!/usr/bin/python3
# coding=utf-8
import numpy as np


def gen_golden_data_simple():
    x = np.random.uniform(-10, 10, [2, 2, 4, 128]).astype(np.float16)
    golden = np.maximum(x, 0).astype(np.float16)
    x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
