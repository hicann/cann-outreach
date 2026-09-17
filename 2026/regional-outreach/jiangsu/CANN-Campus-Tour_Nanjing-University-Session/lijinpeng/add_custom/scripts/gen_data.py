#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 add_custom 算子的输入数据与真值数据。

输入张量：x[N2, N1]、y[N2, N1]，数据类型 float16，数据格式 ND
输出文件：
    input/input_x.bin   输入 x，float16，按行优先（ND）连续排布
    input/input_y.bin   输入 y，float16
    input/golden_z.bin  真值 z = x + y，float16

用法：python3 scripts/gen_data.py [N2] [N1]
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
INPUT_DIR = os.path.join(ROOT_DIR, "input")


def main():
    n2 = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n1 = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    if n2 <= 0 or n1 <= 0:
        print("[ERROR] invalid shape: N2 = {}, N1 = {}".format(n2, n1))
        return 1

    os.makedirs(INPUT_DIR, exist_ok=True)

    # 固定随机种子，保证多次运行结果可复现
    rng = np.random.default_rng(seed=20240916)
    x = rng.uniform(low=-1.0, high=1.0, size=(n2, n1)).astype(np.float16)
    y = rng.uniform(low=-1.0, high=1.0, size=(n2, n1)).astype(np.float16)

    # 真值：按 float32 相加后舍入回 float16（等价于逐元素加法的正确舍入结果）
    golden = (x.astype(np.float32) + y.astype(np.float32)).astype(np.float16)

    x.tofile(os.path.join(INPUT_DIR, "input_x.bin"))
    y.tofile(os.path.join(INPUT_DIR, "input_y.bin"))
    golden.tofile(os.path.join(INPUT_DIR, "golden_z.bin"))

    print("[INFO] shape = [{}, {}], total = {} elements, {} Byte per tensor".format(
        n2, n1, n2 * n1, n2 * n1 * 2))
    print("[INFO] input_x.bin / input_y.bin / golden_z.bin generated in {}".format(INPUT_DIR))
    print("[INFO] x[0][:4]      = {}".format(x.reshape(-1)[:4]))
    print("[INFO] y[0][:4]      = {}".format(y.reshape(-1)[:4]))
    print("[INFO] golden[0][:4] = {}".format(golden.reshape(-1)[:4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
