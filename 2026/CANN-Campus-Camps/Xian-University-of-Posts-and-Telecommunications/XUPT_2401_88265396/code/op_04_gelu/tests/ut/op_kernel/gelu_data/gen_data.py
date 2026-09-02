#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

from scipy.special import erf


def impl(input_x):
    """
    Gelu算子实现
    
    参数:
    input_x: 输入张量
    
    返回:
    output: GELU激活后的张量
    """
    # GELU(x) = x * Φ(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2 = np.sqrt(2.0)
    output = input_x * 0.5 * (1.0 + erf(input_x / sqrt2))
    return output.astype(input_x.dtype)


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
    input_input_x = np.ones((32)).astype(d_type_dict["float16"])

    
    # 计算 golden 数据
    golden = impl(input_input_x)
    
    # 保存数据到文件
    input_input_x.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_gelu_input_x.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            with open("float16_golden_gelu.bin", "wb") as _f:
                for _g in golden:
                    _g.astype(d_type_dict["float16"]).tofile(_f)
        else:
            golden.astype(d_type_dict["float16"]).tofile("float16_golden_gelu.bin")
    
    print(f"生成完成: dtype={d_type}")
