# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def get_np_dtype(d_type):
    if d_type == "int32":
        return np.int32
    if d_type == "int8":
        return np.int8
    if d_type == "uint8":
        return np.uint8
    if d_type == "float16":
        return np.float16
    if d_type == "float32":
        return np.float32
    raise ValueError("unsupported dtype: " + d_type)


def compare_data(golden_file_lists, output_file_lists, d_type):
    np_dtype = get_np_dtype(d_type)

    if len(golden_file_lists) == 0 or len(output_file_lists) == 0:
        print("FAILED!")
        print("golden or output file missing")
        return False

    if len(golden_file_lists) != len(output_file_lists):
        print("FAILED!")
        print("golden and output file counts differ")
        return False

    data_same = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        tmp_out = np.fromfile(out, np_dtype)
        tmp_gold = np.fromfile(gold, np_dtype)

        diff_res = np.isclose(tmp_out, tmp_gold, rtol=1e-3, atol=1e-3)
        diff_idx = np.where(~diff_res)[0]

        if len(diff_idx) == 0:
            print("PASSED!")
        else:
            print("FAILED!")
            for idx in diff_idx[:10]:
                print(f"index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
            data_same = False

    return data_same


def process(d_type):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    result = compare_data(golden_file_lists, output_file_lists, d_type)
    print("compare result:", result)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: compare_data.py <dtype>")
        sys.exit(2)
    ret = process(sys.argv[1])
    sys.exit(0 if ret else 1)
