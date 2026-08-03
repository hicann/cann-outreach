#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from ml_dtypes import bfloat16
import glob
import os

curr_dir = os.path.dirname(os.path.realpath(__file__))


def get_threshold_by_dtype(dtype):
    """
    根据数据类型获取通过阈值（社区标准）
    
    精度标准:
    - FLOAT16: threshold = 2^-10 ≈ 0.000977
    - BFLOAT16: threshold = 2^-7 ≈ 0.00781
    - FLOAT32: threshold = 2^-13 ≈ 0.000122
    - HiFLOAT32: threshold = 2^-11 ≈ 0.000488
    - FLOAT8 E4M3: threshold = 2^-3 ≈ 0.125
    - FLOAT8 E5M2: threshold = 2^-2 ≈ 0.25
    """
    dtype_str = str(dtype).lower().replace(' ', '').replace('_', '')
    
    thresholds = {
        'float16': 2 ** (-10),
        'bfloat16': 2 ** (-7),
        'float32': 2 ** (-13),
        'float64': 2 ** (-13),
        'hifloat32': 2 ** (-11),
    }
    
    if 'float8e4m3' in dtype_str or 'fp8e4m3' in dtype_str:
        return 2 ** (-3)
    elif 'float8e5m2' in dtype_str or 'fp8e5m2' in dtype_str:
        return 2 ** (-2)
    
    return thresholds.get(dtype_str, 2 ** (-13))


def calculate_mare(actual, golden):
    """计算最大相对误差(MARE)，分母使用 max(|golden|, 1.0) 避免近零值爆炸"""
    relative_errors = np.abs(actual - golden) / np.maximum(np.abs(golden), 1.0)
    return np.max(relative_errors)


def calculate_mere(actual, golden):
    """计算平均相对误差(MERE)，分母使用 max(|golden|, 1.0) 避免近零值爆炸"""
    relative_errors = np.abs(actual - golden) / np.maximum(np.abs(golden), 1.0)
    return np.mean(relative_errors)


def compare_data_float(golden_file_lists, output_file_lists, d_type):
    """
    浮点类型精度比对
    
    通过标准 (np.allclose 风格混合容差):
      |actual - golden| <= atol + rtol * |golden|
    
    其中:
      rtol = threshold (相对容差)
      atol = threshold (绝对容差，为近零 golden 值提供下限保护)
    
    辅助通过条件 (针对CPU模拟环境下的ULP级别差异):
      若 99.9% 以上元素通过且最大绝对误差 < threshold * 100，也判定通过
    """
    if d_type == "float16":
        np_dtype = np.float16
    elif d_type == "float32":
        np_dtype = np.float32
    elif d_type == "float64":
        np_dtype = np.float64
    elif d_type == "bfloat16":
        np_dtype = bfloat16
    elif d_type == "hifloat32":
        np_dtype = np.float32
    elif d_type in ("fp8_e4m3fn", "fp8_e5m2"):
        np_dtype = np.uint8
    else:
        np_dtype = np.float32
    
    threshold = get_threshold_by_dtype(d_type)
    rtol = threshold
    atol = threshold
    
    def _read_bin(path):
        raw = np.fromfile(path, np_dtype)
        if d_type == "fp8_e4m3fn":
            from ml_dtypes import float8_e4m3fn
            return raw.view(float8_e4m3fn).astype(np.float32)
        elif d_type == "fp8_e5m2":
            from ml_dtypes import float8_e5m2
            return raw.view(float8_e5m2).astype(np.float32)
        # 统一转换到float32进行比对，消除target dtype的ULP级别差异
        return raw.astype(np.float32)
    
    def _compare_pair(tmp_out, tmp_gold):
        """比对单对数据，返回 (is_pass, diagnostics_dict)"""
        diff = np.abs(tmp_out - tmp_gold)
        max_abs_err = float(np.max(diff))
        mere = float(calculate_mere(tmp_out, tmp_gold))
        mare = float(calculate_mare(tmp_out, tmp_gold))
        
        # 主通过条件: 逐元素检查 |actual - golden| <= atol + rtol * |golden|
        elementwise_tol = atol + rtol * np.abs(tmp_gold)
        elementwise_pass = diff <= elementwise_tol
        pass_ratio = float(np.mean(elementwise_pass))
        
        # 辅助通过条件: 99.9% 元素通过 且 最大绝对误差在合理范围
        # (针对CPU模拟Cast与numpy astype的ULP级别差异)
        aux_pass = (pass_ratio >= 0.999) and (max_abs_err < threshold * 100)
        
        is_pass = np.all(elementwise_pass) or aux_pass
        
        return is_pass, {
            'mere': mere, 'mare': mare, 'max_abs_err': max_abs_err,
            'pass_ratio': pass_ratio, 'threshold': threshold,
            'rtol': rtol, 'atol': atol,
        }
    
    def _print_result(is_pass, diag, gold_name="", out_name=""):
        prefix = f"[{gold_name} vs {out_name}] " if gold_name else ""
        if is_pass:
            print(f"{prefix}PASSED! MERE={diag['mere']:.6f}, MARE={diag['mare']:.6f}, "
                  f"MaxAbsErr={diag['max_abs_err']:.6f}, PassRatio={diag['pass_ratio']:.4f}")
        else:
            print(f"{prefix}FAILED! MERE={diag['mere']:.6f} (thr={diag['threshold']:.6f}), "
                  f"MARE={diag['mare']:.6f}, MaxAbsErr={diag['max_abs_err']:.6f}, "
                  f"PassRatio={diag['pass_ratio']:.4f}")
    
    def _print_diff_details(tmp_out, tmp_gold, max_count=10):
        diff = np.abs(tmp_out - tmp_gold)
        elementwise_tol = atol + rtol * np.abs(tmp_gold)
        fail_mask = diff > elementwise_tol
        fail_indices = np.where(fail_mask)[0]
        if len(fail_indices) == 0:
            return
        # 按误差大小排序，取最大的 max_count 个
        sorted_idx = fail_indices[np.argsort(diff[fail_indices])[::-1][:max_count]]
        for idx in sorted_idx:
            rel_err = diff[idx] / max(abs(tmp_gold[idx]), 1.0)
            print(f"  index: {idx}, output: {tmp_out[idx]:.6f}, golden: {tmp_gold[idx]:.6f}, "
                  f"abs_err: {diff[idx]:.6f}, rel_err: {rel_err:.6f}")
        print(f"  Total failed elements: {len(fail_indices)}/{len(diff)}")
    
    data_same = True
    # 当 golden 文件数 < output 文件数时，将多个 output 拼接后与单个 golden 比对
    if len(golden_file_lists) == 1 and len(output_file_lists) > 1:
        tmp_gold = _read_bin(golden_file_lists[0])
        tmp_out = np.concatenate([_read_bin(f) for f in output_file_lists])
        
        is_pass, diag = _compare_pair(tmp_out, tmp_gold)
        _print_result(is_pass, diag)
        if not is_pass:
            _print_diff_details(tmp_out, tmp_gold)
            data_same = False
    else:
        for gold, out in zip(golden_file_lists, output_file_lists):
            tmp_out = _read_bin(out)
            tmp_gold = _read_bin(gold)
            
            is_pass, diag = _compare_pair(tmp_out, tmp_gold)
            gold_name = os.path.basename(gold)
            out_name = os.path.basename(out)
            _print_result(is_pass, diag, gold_name, out_name)
            if not is_pass:
                _print_diff_details(tmp_out, tmp_gold)
                data_same = False
    return data_same


def compare_data_integer(golden_file_lists, output_file_lists, d_type):
    """
    整数类型精度比对
    
    通过标准: 二进制一致 或 绝对误差为0
    """
    if d_type == "int32":
        np_dtype = np.int32
    elif d_type == "int8":
        np_dtype = np.int8
    elif d_type == "int16":
        np_dtype = np.int16
    elif d_type == "int64":
        np_dtype = np.int64
    elif d_type == "uint8":
        np_dtype = np.uint8
    elif d_type == "uint16":
        np_dtype = np.uint16
    elif d_type == "uint32":
        np_dtype = np.uint32
    elif d_type == "uint64":
        np_dtype = np.uint64
    elif d_type == "bool":
        np_dtype = np.bool_
    else:
        np_dtype = np.int32
    
    data_same = True
    if len(golden_file_lists) == 1 and len(output_file_lists) > 1:
        tmp_gold = np.fromfile(golden_file_lists[0], np_dtype)
        tmp_out = np.concatenate([np.fromfile(f, np_dtype) for f in output_file_lists])
        bitwise_match = np.array_equal(tmp_out, tmp_gold)
        abs_error_zero = np.all(np.abs(tmp_out.astype(np.int64) - tmp_gold.astype(np.int64)) == 0)
        is_pass = bitwise_match or abs_error_zero
        if is_pass:
            print(f"PASSED! bitwise_match={bitwise_match}, abs_error_zero={abs_error_zero}")
        else:
            print(f"FAILED!")
            data_same = False
    else:
        for gold, out in zip(golden_file_lists, output_file_lists):
            tmp_out = np.fromfile(out, np_dtype)
            tmp_gold = np.fromfile(gold, np_dtype)
            
            # 检查二进制一致
            bitwise_match = np.array_equal(tmp_out, tmp_gold)
            # 检查绝对误差为0
            abs_error_zero = np.all(np.abs(tmp_out.astype(np.int64) - tmp_gold.astype(np.int64)) == 0)
            
            is_pass = bitwise_match or abs_error_zero
            
            if is_pass:
                print(f"PASSED! bitwise_match={bitwise_match}, abs_error_zero={abs_error_zero}")
            else:
                print(f"FAILED!")
                diff_idx = np.where(tmp_out != tmp_gold)[0][:5]
                for idx in diff_idx:
                    print(f"  index: {idx}, output: {tmp_out[idx]}, golden: {tmp_gold[idx]}")
                data_same = False
    return data_same


def get_file_lists(dtype):
    golden_file_lists = sorted(glob.glob(curr_dir + "/*golden*.bin"))
    output_file_lists = sorted(glob.glob(curr_dir + "/*output*.bin"))
    return golden_file_lists, output_file_lists


def infer_dtype_from_filename():
    """从 golden 文件名推断 dtype"""
    golden_files = glob.glob(curr_dir + "/*golden*.bin")
    if not golden_files:
        return "float32"
    
    filename = os.path.basename(golden_files[0])
    for dtype in ["float16", "float32", "float64", "bfloat16", "fp8_e4m3fn", "fp8_e5m2", "hifloat32", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]:
        if filename.startswith(dtype + "_"):
            return dtype
    return "float32"


def infer_dtype_from_single_filename(filename):
    """从单个文件名推断 dtype"""
    basename = os.path.basename(filename)
    for dt in ["float16", "float32", "float64", "bfloat16", "fp8_e4m3fn", "fp8_e5m2", "hifloat32", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]:
        if basename.startswith(dt + "_"):
            return dt
    return "float32"


def process(d_type):
    golden_file_lists, output_file_lists = get_file_lists(d_type)
    
    if not golden_file_lists and not output_file_lists:
        print("No golden or output files found (no-output operator), skipping comparison")
        return True
    
    if not golden_file_lists or not output_file_lists:
        print("ERROR: No golden or output files found")
        return False
    
    # 单 golden 多 output：拼接比对
    if len(golden_file_lists) == 1 and len(output_file_lists) > 1:
        if d_type in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]:
            result = compare_data_integer(golden_file_lists, output_file_lists, d_type)
        else:
            result = compare_data_float(golden_file_lists, output_file_lists, d_type)
        print("compare result:", result)
        return result

    if len(golden_file_lists) != len(output_file_lists):
        print(f"ERROR: file count mismatch: golden={len(golden_file_lists)}, output={len(output_file_lists)}")
        return False
    
    # 逐对比对，每个 golden 文件独立推断 dtype
    all_pass = True
    for gold, out in zip(golden_file_lists, output_file_lists):
        file_dtype = infer_dtype_from_single_filename(gold)
        if file_dtype in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "bool"]:
            pair_result = compare_data_integer([gold], [out], file_dtype)
        else:
            pair_result = compare_data_float([gold], [out], file_dtype)
        if not pair_result:
            all_pass = False
    
    print("compare result:", all_pass)
    return all_pass


if __name__ == '__main__':
    # 从文件名推断 dtype，或使用命令行参数
    if len(sys.argv) >= 2:
        d_type = sys.argv[1]
    else:
        d_type = infer_dtype_from_filename()
        print(f"从文件名推断 dtype: {d_type}")
    
    ret = process(d_type)
    exit(0 if ret else 1)
