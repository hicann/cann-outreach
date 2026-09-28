#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gcd123 算子结果校验脚本。

对比算子实际输出 out.bin 与 golden.bin（fp16）：
  - 输入数据为 fp16 精确表示的整数，结果应完全一致（允许 0.5 容差做稳健判断）。

用法：
  python3 verify_result.py --out ./st/out/test_data/output/out.bin --golden ./test_data/golden.bin
  python3 verify_result.py --outdir ./test_data/case1          # 对比 case1 的 out.bin / golden.bin
"""
import argparse
import os
import sys

import numpy as np


def load_bin(path):
    if not os.path.exists(path):
        print("[verify] ERROR: file not found: %s" % path)
        sys.exit(1)
    return np.fromfile(path, dtype=np.float16)


def main():
    parser = argparse.ArgumentParser(description="gcd123 result verifier")
    parser.add_argument("--out", type=str, default=None, help="算子实际输出文件")
    parser.add_argument("--golden", type=str, default=None, help="黄金输出文件")
    parser.add_argument("--outdir", type=str, default=None,
                        help="目录模式：对比 <outdir>/out.bin 与 <outdir>/golden.bin")
    args = parser.parse_args()

    if args.outdir:
        args.out = os.path.join(args.outdir, "out.bin")
        args.golden = os.path.join(args.outdir, "golden.bin")
    if not args.out or not args.golden:
        print("[verify] ERROR: need --out/--golden or --outdir")
        sys.exit(1)

    out_arr = load_bin(args.out)
    golden_arr = load_bin(args.golden)
    if out_arr.shape[0] != golden_arr.shape[0]:
        print("[verify] FAIL: size mismatch, out=%d golden=%d"
              % (out_arr.shape[0], golden_arr.shape[0]))
        sys.exit(1)

    diff = np.abs(out_arr.astype(np.float64) - golden_arr.astype(np.float64))
    # 整数数据应完全一致；给 0.5 容差以容忍极端舍入情形
    bad = int(np.sum(diff > 0.5))
    if bad == 0:
        print("[verify] PASS: %d elements, all match golden" % out_arr.shape[0])
        sys.exit(0)
    print("[verify] FAIL: %d / %d elements mismatch (max diff=%f)"
          % (bad, out_arr.shape[0], float(diff.max())))
    sys.exit(1)


if __name__ == "__main__":
    main()
