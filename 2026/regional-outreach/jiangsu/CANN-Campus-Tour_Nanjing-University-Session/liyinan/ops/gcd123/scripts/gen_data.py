#!/usr/bin/python3
# coding=utf-8
"""
gcd123 data generator with broadcast.
self  [1, 4, 1, 128]
other [1, 4, 16, 1]
out   [1, 4, 16, 128]
Host materializes broadcast so kernel sees equal-length tensors.
"""
import numpy as np


def gcd_trunc(a, b):
    ai = np.abs(np.trunc(a.astype(np.float32))).astype(np.int32)
    bi = np.abs(np.trunc(b.astype(np.float32))).astype(np.int32)
    return np.gcd(ai, bi).astype(np.float16)


def gen_golden_data_simple():
    self_raw = np.random.randint(1, 50, size=[1, 4, 1, 128]).astype(np.float16)
    other_raw = np.random.randint(1, 50, size=[1, 4, 16, 1]).astype(np.float16)

    self_b = np.broadcast_to(self_raw, [1, 4, 16, 128]).copy()
    other_b = np.broadcast_to(other_raw, [1, 4, 16, 128]).copy()
    golden = gcd_trunc(self_b, other_b)

    self_b.tofile("./input/input_self.bin")
    other_b.tofile("./input/input_other.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
