#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gcd123 算子测试数据生成脚本。

为每组用例生成：
  - self.bin / other.bin ：输入数据（fp16，ND 格式，4 维）
  - golden.bin           ：黄金输出（与 kernel 语义一致：整数域上的最大公约数）

数据取值：0 ~ 200 之间的整数（fp16 可精确表示），并有意包含 0，以覆盖
gcd(a, 0) = |a|、gcd(0, 0) = 0 的边界情形。

用法：
  python3 gen_data.py                        # 生成全部内置用例
  python3 gen_data.py --case 1               # 只生成 case1
  python3 gen_data.py --self 1,3,4,1 --other 2,1,4,8 --outdir ./test_data
"""
import argparse
import math
import os
import sys

import numpy as np

# 内置用例（覆盖各类 broadcast 与切分路径）
CASES = [
    # (self_shape, other_shape, 说明)
    ([1, 3, 4, 1], [2, 1, 4, 8],  "case1: 双端 broadcast, out=[2,3,4,8]"),
    ([2, 2, 2, 16], [2, 2, 2, 16], "case2: shape 完全一致"),
    ([1, 1, 1, 4096], [1, 1, 1, 1], "case3: N1>2048 行内切块 + 标量 broadcast"),
    ([1, 1, 1, 1], [2, 3, 4, 8],  "case4: self 为标量(整 tile 重复)"),
    ([2, 1, 1, 8], [1, 3, 4, 1],  "case5: 覆盖 s3==0 行值广播与 s2==0 整行重复路径"),
    ([1, 1, 1, 63], [1, 1, 1, 63], "case6: N1 非 16 对齐(行尾填充路径)"),
    ([1, 1, 2, 3], [2, 3, 1, 1],   "case7: 小 shape 混合 broadcast, out=[2,3,2,3]"),
]


def _broadcast_out_shape(shape_a, shape_b):
    out = []
    for a, b in zip(shape_a, shape_b):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            raise ValueError("shapes are not broadcastable: %s vs %s" % (shape_a, shape_b))
    return list(out)


def _int_gcd(a, b):
    a = abs(int(a))
    b = abs(int(b))
    while b != 0:
        a, b = b, a % b
    return a


def gen_case(self_shape, other_shape, outdir, seed=42, value_max=200):
    rng = np.random.default_rng(seed)
    out_shape = _broadcast_out_shape(self_shape, other_shape)

    # 输入为整数（fp16 精确表示范围内），并混入 0 值
    self_arr = rng.integers(0, value_max + 1, size=self_shape).astype(np.float64)
    other_arr = rng.integers(0, value_max + 1, size=other_shape).astype(np.float64)

    # 黄金输出：broadcast 后逐元素整数 gcd
    self_b = np.broadcast_to(self_arr, out_shape)
    other_b = np.broadcast_to(other_arr, out_shape)
    golden = np.empty(out_shape, dtype=np.float64)
    it = np.nditer([self_b, other_b], flags=["multi_index"])
    for s, o in it:
        golden[it.multi_index] = _int_gcd(round(float(s)), round(float(o)))

    os.makedirs(outdir, exist_ok=True)
    self_arr.astype(np.float16).tofile(os.path.join(outdir, "self.bin"))
    other_arr.astype(np.float16).tofile(os.path.join(outdir, "other.bin"))
    golden.astype(np.float16).tofile(os.path.join(outdir, "golden.bin"))

    # 用例信息文件（shape 描述，供 ST 用例 json 参考）
    with open(os.path.join(outdir, "shape_info.txt"), "w") as f:
        f.write("self_shape=%s\n" % list(self_shape))
        f.write("other_shape=%s\n" % list(other_shape))
        f.write("out_shape=%s\n" % out_shape)

    print("[gen] %s self=%s other=%s -> out=%s"
          % (outdir, list(self_shape), list(other_shape), out_shape))
    return out_shape


def main():
    parser = argparse.ArgumentParser(description="gcd123 test data generator")
    parser.add_argument("--case", type=int, default=0, help="内置用例编号(1..N)，0 表示全部")
    parser.add_argument("--self", type=str, default=None, help="self 的 shape，如 1,3,4,1")
    parser.add_argument("--other", type=str, default=None, help="other 的 shape，如 2,1,4,8")
    parser.add_argument("--outdir", type=str, default="./test_data", help="输出目录")
    args = parser.parse_args()

    if args.self is not None and args.other is not None:
        self_shape = [int(x) for x in args.self.split(",")]
        other_shape = [int(x) for x in args.other.split(",")]
        if len(self_shape) != 4 or len(other_shape) != 4:
            print("[gen] ERROR: shape must be 4-D, e.g. 1,3,4,1")
            sys.exit(1)
        gen_case(self_shape, other_shape, args.outdir)
    elif args.case > 0:
        if args.case > len(CASES):
            print("[gen] ERROR: case %d not found" % args.case)
            sys.exit(1)
        shape_s, shape_o, desc = CASES[args.case - 1]
        print("[gen] %s" % desc)
        gen_case(shape_s, shape_o, os.path.join(args.outdir, "case%d" % args.case))
    else:
        for idx, (shape_s, shape_o, desc) in enumerate(CASES):
            print("[gen] %s" % desc)
            gen_case(shape_s, shape_o, os.path.join(args.outdir, "case%d" % (idx + 1)))


if __name__ == "__main__":
    main()
