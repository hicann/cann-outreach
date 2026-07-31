#!/usr/bin/env python3
"""Precision comparison: kernel output vs golden for each dtype"""
import os, glob
import numpy as np

def read_bf16_as_f32(path):
    """Read bfloat16 bin, convert to float32 by left-shifting"""
    u16 = np.fromfile(path, np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32)

def compare(dt):
    gold_files = sorted(glob.glob(f"{dt}_golden*.bin"))
    out_files  = sorted(glob.glob(f"{dt}_output*.bin"))
    if not gold_files or not out_files:
        print(f"[SKIP] {dt}: no files"); return True

    if dt == "bfloat16":
        golden = np.concatenate([np.fromfile(f, np.float32) for f in gold_files])
        output = np.concatenate([read_bf16_as_f32(f) for f in out_files])
    elif "int" in dt:
        np_t = {"int32":np.int32,"int8":np.int8,"uint8":np.uint8}[dt]
        golden = np.concatenate([np.fromfile(f, np_t) for f in gold_files])
        output = np.concatenate([np.fromfile(f, np_t) for f in out_files])
    else:
        np_t = {"float16":np.float16,"float32":np.float32}[dt]
        golden = np.concatenate([np.fromfile(f, np_t) for f in gold_files])
        output = np.concatenate([np.fromfile(f, np_t) for f in out_files])

    if golden.shape != output.shape:
        print(f"[FAIL] {dt}: shape mismatch {golden.shape} vs {output.shape}")
        return False

    if "float" in dt or dt == "bfloat16":
        gf, of = np.float32(golden), np.float32(output)
        diff = np.abs(gf - of)
        # Use per-element relative error, with atol for near-zero
        denom = np.maximum(np.abs(gf), 1e-6)
        mere = np.mean(diff / denom)
        mare = np.max(diff / denom)
        ok = mare < 0.02  # 2% for bf16 tolerance
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {dt}: MERE={mere:.6f} MARE={mare:.6f}")
        if not ok:
            idx = np.argsort(diff)[-3:]
            for i in idx:
                print(f"  [{i}] golden={gf[i]:.6f} output={of[i]:.6f}")
    else:
        ok = np.array_equal(golden, output)
        tag = "PASS" if ok else "FAIL"
        print(f"[{tag}] {dt}: bitwise {'match' if ok else 'mismatch'}")
        if not ok:
            bad = np.where(golden != output)[0]
            for i in bad[:3]:
                print(f"  [{i}] golden={golden[i]} output={output[i]}")
    return ok

if __name__ == "__main__":
    all_ok = True
    for dt in ["float16","float32","bfloat16","int32","int8","uint8"]:
        if not compare(dt): all_ok = False
    print(f"\n{'ALL PASSED' if all_ok else 'SOME FAILED'}")
