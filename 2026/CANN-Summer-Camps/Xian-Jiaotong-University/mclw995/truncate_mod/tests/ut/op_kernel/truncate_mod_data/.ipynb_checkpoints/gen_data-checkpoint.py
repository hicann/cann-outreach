#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

def impl(x1, x2):
    return x1 - np.trunc(x1 / x2) * x2

if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "float16"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]
    
    # 生成输入数据
    input_x1 = np.ones((5)).astype(d_type_dict["float16"])
    input_x2 = np.ones((5)).astype(d_type_dict["float16"])

    
    # 计算 golden 数据
    golden = impl(input_x1, input_x2)
    
    # 保存数据到文件
    input_x1.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_truncate_mod_x1.bin")
    input_x2.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_truncate_mod_x2.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float16_golden_truncate_mod.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float16"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float16"]).tofile("float16_golden_truncate_mod.bin")
    
    print(f"生成完成: dtype={d_type}")
