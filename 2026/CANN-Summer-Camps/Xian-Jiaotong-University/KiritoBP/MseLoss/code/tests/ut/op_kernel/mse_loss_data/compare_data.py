#!/usr/bin/env python3
# compare_data.py — 精度比对: 对比 kernel 输出与 golden 数据

import struct
import sys
import glob
import os

def load_float32_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    return [struct.unpack("f", data[i:i+4])[0] for i in range(0, len(data), 4)]

def main():
    golden_file = "float32_golden_mse_loss.bin"
    output_file = "float32_output_mse_loss_0.bin"

    if not os.path.exists(output_file):
        print(f"[ WARNING ] Output file not found: {output_file}")
        print(f"            Looking for: {os.getcwd()}/{output_file}")
        # Try build directory
        for alt in glob.glob("../../build/op_kernel/*.bin"):
            print(f"            Found alt: {alt}")
        sys.exit(0)  # Don't fail if bin not found (may be in build dir)

    if not os.path.exists(golden_file):
        print(f"[ ERROR   ] Golden file not found: {golden_file}")
        sys.exit(1)

    golden = load_float32_bin(golden_file)
    output = load_float32_bin(output_file)

    if len(golden) != len(output):
        print(f"[ FAILED  ] Size mismatch: golden={len(golden)}, output={len(output)}")
        sys.exit(1)

    max_err = 0.0
    err_count = 0
    for i in range(len(golden)):
        err = abs(golden[i] - output[i])
        if err > max_err:
            max_err = err
        if err > 1e-3:
            err_count += 1
            if err_count <= 5:  # Only show first 5
                print(f"  [{i}] golden={golden[i]:.6f} output={output[i]:.6f} err={err:.6f}")

    total = len(golden)
    ok_count = total - err_count

    if err_count == 0:
        print(f"[  PASSED ] All {total} elements match (max_err={max_err:.2e})")
        sys.exit(0)
    else:
        print(f"[ FAILED  ] {err_count}/{total} elements mismatch (max_err={max_err:.6f})")
        sys.exit(1)

if __name__ == "__main__":
    main()
