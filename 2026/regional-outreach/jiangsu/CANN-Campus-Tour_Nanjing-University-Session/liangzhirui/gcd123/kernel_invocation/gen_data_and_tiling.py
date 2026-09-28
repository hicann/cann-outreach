#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gcd123 直接调用（kernel_invocation）模式的数据与 tiling 生成脚本。

生成：
  input/self.bin   - self 输入（fp16，4 维 ND）
  input/other.bin  - other 输入（fp16，4 维 ND）
  input/tiling.bin - tiling 数据（19 个 uint32 字段，顺序与 gcd123_tiling_def.h 一致）
  output/golden.bin- 黄金输出

用法：
  python3 gen_data_and_tiling.py                          # 默认 [1,3,4,1] x [2,1,4,8]
  python3 gen_data_and_tiling.py --self 1,1,1,4096 --other 1,1,1,1
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import gen_data as gd  # noqa: E402  复用内置用例的 gcd/broadcast 计算

CHUNK = 2048
MAX_BLOCK_DIM = 8
GCD_MAX_ITER = 64

# tiling 结构体字段顺序（与 kernel_invocation/gcd123_tiling_def.h 保持一致）
TILING_FIELDS = [
    "blockDim", "totalLength", "coreLen", "tileSize", "rowsPerTile", "pitch", "maxIter",
    "n4", "n3", "n2", "n1",
    "selfS0", "selfS1", "selfS2", "selfS3",
    "otherS0", "otherS1", "otherS2", "otherS3",
]


def compute_tiling(self_shape, other_shape):
    """与 op_host/gcd123.cpp 中 TilingFunc 的切分逻辑保持一致。"""
    out_shape = gd._broadcast_out_shape(self_shape, other_shape)
    n4, n3, n2, n1 = out_shape
    total = n4 * n3 * n2 * n1

    strides = []
    for sh in (self_shape, other_shape):
        s = [0] * 4
        acc = 1
        for d in range(3, -1, -1):
            s[d] = 0 if sh[d] == 1 else acc
            acc *= sh[d]
        strides.append(s)

    pitch = ((n1 + 15) // 16) * 16
    if n1 < CHUNK:
        rows_per_tile = max(1, min(CHUNK // pitch, n2))
        tile_size = rows_per_tile * pitch
        real_per_tile = rows_per_tile * n1
    else:
        rows_per_tile = 0
        tile_size = CHUNK
        real_per_tile = CHUNK

    tiles_count = (total + real_per_tile - 1) // real_per_tile
    block_dim = min(MAX_BLOCK_DIM, max(1, (total + CHUNK - 1) // CHUNK))
    core_len = ((tiles_count + block_dim - 1) // block_dim) * real_per_tile

    return {
        "blockDim": block_dim,
        "totalLength": total,
        "coreLen": core_len,
        "tileSize": tile_size,
        "rowsPerTile": rows_per_tile,
        "pitch": pitch,
        "maxIter": GCD_MAX_ITER,
        "n4": n4, "n3": n3, "n2": n2, "n1": n1,
        "selfS0": strides[0][0], "selfS1": strides[0][1],
        "selfS2": strides[0][2], "selfS3": strides[0][3],
        "otherS0": strides[1][0], "otherS1": strides[1][1],
        "otherS2": strides[1][2], "otherS3": strides[1][3],
    }


def main():
    parser = argparse.ArgumentParser(description="gcd123 kernel_invocation data/tiling generator")
    parser.add_argument("--self", type=str, default="1,3,4,1", help="self 的 shape（4 维）")
    parser.add_argument("--other", type=str, default="2,1,4,8", help="other 的 shape（4 维）")
    parser.add_argument("--value-max", type=int, default=200, help="随机整数上限（fp16 精确范围 2048 内）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    self_shape = [int(x) for x in args.self.split(",")]
    other_shape = [int(x) for x in args.other.split(",")]
    if len(self_shape) != 4 or len(other_shape) != 4:
        print("[gen] ERROR: shape must be 4-D, e.g. 1,3,4,1")
        sys.exit(1)

    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    # 数据 + 黄金输出
    rng = np.random.default_rng(args.seed)
    out_shape = gd._broadcast_out_shape(self_shape, other_shape)
    self_arr = rng.integers(0, args.value_max + 1, size=self_shape).astype(np.float64)
    other_arr = rng.integers(0, args.value_max + 1, size=other_shape).astype(np.float64)
    self_b = np.broadcast_to(self_arr, out_shape)
    other_b = np.broadcast_to(other_arr, out_shape)
    golden = np.empty(out_shape, dtype=np.float64)
    it = np.nditer([self_b, other_b], flags=["multi_index"])
    for s, o in it:
        golden[it.multi_index] = gd._int_gcd(round(float(s)), round(float(o)))

    self_arr.astype(np.float16).tofile("./input/self.bin")
    other_arr.astype(np.float16).tofile("./input/other.bin")
    golden.astype(np.float16).tofile("./output/golden.bin")

    # tiling 数据（uint32 小端，字段顺序与 C++ 结构体一致）
    tiling = compute_tiling(self_shape, other_shape)
    with open("./input/tiling.bin", "wb") as f:
        for name in TILING_FIELDS:
            f.write(struct.pack("<I", tiling[name]))

    print("[gen] self=%s other=%s -> out=%s" % (self_shape, other_shape, out_shape))
    print("[gen] tiling: blockDim=%d totalLength=%d coreLen=%d tileSize=%d "
          "rowsPerTile=%d pitch=%d maxIter=%d" %
          (tiling["blockDim"], tiling["totalLength"], tiling["coreLen"],
           tiling["tileSize"], tiling["rowsPerTile"], tiling["pitch"], tiling["maxIter"]))


if __name__ == "__main__":
    main()
