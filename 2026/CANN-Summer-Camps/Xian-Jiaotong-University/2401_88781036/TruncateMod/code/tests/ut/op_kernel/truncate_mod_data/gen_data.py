# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import sys
import glob
import numpy as np


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip().strip("(").strip(")")
    shape_list = [int(x.strip()) for x in shape_str.split(",") if x.strip()]
    return tuple(shape_list)


def gen_data_and_golden(shape_str, d_type="int32"):
    d_type_dict = {
        "int32": np.int32,
        "int8": np.int8,
        "uint8": np.uint8,
        "float16": np.float16,
        "float32": np.float32,
    }

    if d_type not in d_type_dict:
        raise ValueError("unsupported dtype: " + d_type)

    np_type = d_type_dict[d_type]
    shape = parse_str_to_shape_list(shape_str)

    np.random.seed(0)

    if d_type == "uint8":
        input_x1 = np.random.randint(0, 20, shape).astype(np_type)
        input_x2 = np.random.randint(1, 8, shape).astype(np_type)
    elif d_type in ("int32", "int8"):
        input_x1 = np.random.randint(-20, 20, shape).astype(np_type)
        input_x2 = np.random.randint(-8, 8, shape)
        input_x2[input_x2 == 0] = 3
        input_x2 = input_x2.astype(np_type)
    else:
        input_x1 = np.random.uniform(-20, 20, shape).astype(np_type)
        input_x2 = np.random.uniform(-8, 8, shape)
        input_x2[np.abs(input_x2) < 0.1] = 3.0
        input_x2 = input_x2.astype(np_type)

    # truncate_mod: y = x1 - trunc(x1 / x2) * x2
    golden = np.fmod(input_x1.astype(np.float64), input_x2.astype(np.float64)).astype(
        np_type
    )

    input_x1.tofile(f"{d_type}_input_t1_truncate_mod.bin")
    input_x2.tofile(f"{d_type}_input_t2_truncate_mod.bin")
    golden.tofile(f"{d_type}_golden_t_truncate_mod.bin")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Param num must be 3.")
        exit(1)

    for file_name in glob.glob("*.bin"):
        os.remove(file_name)
    gen_data_and_golden(sys.argv[1], sys.argv[2])
