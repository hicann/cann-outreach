#!/usr/bin/env python3
# gen_data.py — 生成 MseLoss Kernel UT 测试 golden 数据
# predict=1.0, label=2.0 → golden=(1-2)^2=1.0

import struct
import os

def gen_float32_data():
    size = 8 * 2048  # total elements

    # predict: all 1.0
    with open("float32_input_predict.bin", "wb") as f:
        for _ in range(size):
            f.write(struct.pack("f", 1.0))

    # label: all 2.0
    with open("float32_input_label.bin", "wb") as f:
        for _ in range(size):
            f.write(struct.pack("f", 2.0))

    # golden: all 1.0 = (1-2)^2
    with open("float32_golden_mse_loss.bin", "wb") as f:
        for _ in range(size):
            f.write(struct.pack("f", 1.0))

    print(f"[  DATA   ] Generated float32 test data ({size} elements)")

if __name__ == "__main__":
    gen_float32_data()
