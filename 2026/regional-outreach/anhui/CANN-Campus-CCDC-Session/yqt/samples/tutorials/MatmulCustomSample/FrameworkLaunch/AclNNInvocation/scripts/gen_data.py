#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import os


def gen_golden_data():
    M = 1024
    N = 640
    K = 256

    input_a = np.random.randint(1, 10, [M, K]).astype(np.float16)
    input_b = np.random.randint(1, 10, [K, N]).astype(np.float16)
    input_bias = np.random.randint(1, 10, [N]).astype(np.float32)
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)

    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("output"):
        os.mkdir("output")
    input_a.tofile("./input/input_a.bin")
    input_b.tofile("./input/input_b.bin")
    input_bias.tofile("./input/input_bias.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
