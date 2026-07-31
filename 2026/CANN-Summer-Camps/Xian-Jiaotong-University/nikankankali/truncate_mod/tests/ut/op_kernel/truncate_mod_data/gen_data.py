#!/usr/bin/env python3
"""TruncateMod test data — 6 dtypes, 32-byte aligned, proper bf16/int32 handling"""
import os, glob
import numpy as np

ALIGN = {"float16":16, "float32":64, "bfloat16":16, "int32":256, "int8":32, "uint8":32}

def to_bf16(arr_f32):
    """float32 → bfloat16: keep upper 16 bits as uint16"""
    return (arr_f32.view(np.uint32) >> 16).astype(np.uint16)

def from_bf16(arr_u16):
    """bfloat16 → float32: pad lower 16 bits with zeros"""
    return (arr_u16.astype(np.uint32) << 16).view(np.float32)

def golden(x1, x2):
    x1f = np.float32(x1); x2f = np.float32(x2)
    return x1f - np.trunc(x1f / x2f) * x2f

if __name__ == "__main__":
    for f in glob.glob("*_input_*.bin"): os.remove(f)
    for f in glob.glob("*_golden_*.bin"): os.remove(f)
    for f in glob.glob("*_output_*.bin"): os.remove(f)

    for dt in ["float16","float32","bfloat16","int32","int8","uint8"]:
        N = ALIGN[dt]
        if dt == "bfloat16":
            x1_f32 = np.random.randn(N).astype(np.float32) * 5
            x2_f32 = (np.random.randn(N) + 2.0).astype(np.float32)
            x2_f32[np.abs(x2_f32) < 0.1] = 1.0
            to_bf16(x1_f32).tofile(f"{dt}_input_truncate_mod_x1.bin")
            to_bf16(x2_f32).tofile(f"{dt}_input_truncate_mod_x2.bin")
            # Golden on the exact bf16-representable values
            g = golden(from_bf16(to_bf16(x1_f32)), from_bf16(to_bf16(x2_f32)))
        elif "int" in dt:
            np_t = {"int32":np.int32,"int8":np.int8,"uint8":np.uint8}[dt]
            x1 = np.random.randint(-10,11,N).astype(np_t)
            x2 = np.random.randint(-10,11,N).astype(np_t); x2[x2==0]=1
            x1.tofile(f"{dt}_input_truncate_mod_x1.bin")
            x2.tofile(f"{dt}_input_truncate_mod_x2.bin")
            g = golden(x1, x2).astype(np_t)
        else:
            np_t = {"float16":np.float16,"float32":np.float32}[dt]
            x1 = (np.random.randn(N)*5).astype(np_t)
            x2 = (np.random.randn(N)+2.0).astype(np_t); x2[np.abs(x2)<0.1]=1.0
            x1.tofile(f"{dt}_input_truncate_mod_x1.bin")
            x2.tofile(f"{dt}_input_truncate_mod_x2.bin")
            g = golden(x1, x2).astype(np_t)
        g.tofile(f"{dt}_golden_truncate_mod.bin")
        print(f"[OK] {dt:12s}  N={N:3d}")
    print(f"\n6 dtypes done")
