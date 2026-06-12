#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# gen_data.py — 生成可执行文件通路的测试数据

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden import generate_data, generate_test_cases


def save_binary(filename, data: np.ndarray):
    data.astype(np.float16).tofile(filename)


def main():
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test")
    os.makedirs(output_dir, exist_ok=True)

    print("生成 ReLU 测试数据...")
    for name, shape, data_type in generate_test_cases():
        x, y = generate_data(name, shape, data_type)
        save_binary(os.path.join(output_dir, f"{name}_x.bin"), x.numpy())
        save_binary(os.path.join(output_dir, f"{name}_golden.bin"), y.numpy())

        info_path = os.path.join(output_dir, f"{name}_info.txt")
        with open(info_path, "w") as f:
            f.write(f"shape={shape[0]},{shape[1]},{shape[2]},{shape[3]}\n")
            f.write(f"elements={shape[0]*shape[1]*shape[2]*shape[3]}\n")
            f.write("dtype=float16\n")

        print(f"  {name}: shape={shape}")

    print("完成！")


if __name__ == "__main__":
    main()
