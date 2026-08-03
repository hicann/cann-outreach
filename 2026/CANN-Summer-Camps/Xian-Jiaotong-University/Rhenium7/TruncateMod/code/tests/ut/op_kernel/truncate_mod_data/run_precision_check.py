#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precision acceptance test for truncate_mod operator.
Runs mixed_tolerance_check with IEEE special value handling on all test cases and generates summary.txt.
"""

import sys
import os
import numpy as np
from ml_dtypes import bfloat16

# Add the skill scripts to path
sys.path.insert(0, '/mnt/workspace/code/.opencode/skills/ops-precision-standard/scripts')
from mixed_tolerance_check import get_tolerance_by_dtype, _ulp_at_one

# Data directory
DATA_DIR = '/mnt/workspace/code/tests/ut/op_kernel/truncate_mod_data'

# Map case ID to (dtype, shape, description)
# Based on gen_data.py cases K1-K10
CASES = [
    # FP16 cases
    ("float16_K1", np.float16, (5,), "K1: FP16 ELEWISE small (5 elements)"),
    ("float16_K2", np.float16, (8192,), "K2: FP16 ELEWISE multi-core (8192 elements)"),
    ("float16_K6", np.float16, (1024,), "K6: FP16 scalar broadcast (1024 elements)"),
    ("float16_K9_fp16", np.float16, (1003,), "K9: FP16 unaligned tile + tail (1003 elements)"),
    
    # BF16 cases
    ("bfloat16_K3", bfloat16, (8192,), "K3: BF16 ELEWISE multi-core (8192 elements)"),
    ("bfloat16_K9_bf16", bfloat16, (1003,), "K9: BF16 unaligned tile + tail (1003 elements)"),
    ("bfloat16_K10", bfloat16, (4, 5, 32), "K10: BF16 left prefix broadcast (640 elements)"),
    
    # FP32 cases
    ("float32_K4", np.float32, (8192,), "K4: FP32 ELEWISE multi-core (8192 elements)"),
    ("float32_K5", np.float32, (64, 64), "K5: FP32 row broadcast (4096 elements)"),
    ("float32_K7", np.float32, (2048,), "K7: FP32 mixed sign + zero divisor (2048 elements)"),
    ("float32_K8", np.float32, (1024,), "K8: FP32 large quotient (1024 elements)"),
    ("float32_K9_fp32", np.float32, (1003,), "K9: FP32 unaligned tile + tail (1003 elements)"),
]

_ULP_FACTOR = 32


def read_bin_file(filename, dtype):
    """Read binary file and return numpy array."""
    if dtype == bfloat16:
        return np.fromfile(filename, dtype=bfloat16)
    return np.fromfile(filename, dtype=dtype)


def check_mixed_tolerance_ieee(npu_output, golden_output):
    """
    检查浮点算子精度(混合容差 atol/rtol 标准)，支持 IEEE 特殊值处理
    
    通过标准(同时满足):
    1. matched_ratio >= required_matched_ratio
    2. max_abs_error <= max_abs_error_limit
       max_abs_error_limit = max(fixed_limit, 32 * ULP)
    
    特殊值处理(IEEE 语义):
    - NaN == NaN 视为匹配
    - +Inf == +Inf 视为匹配
    - -Inf == -Inf 视为匹配
    - 有限值与特殊值不匹配
    - 特殊值之间异号不匹配
    """
    # Determine dtype from npu_output
    dtype = npu_output.dtype
    rtol, atol, required_matched_ratio, fixed_limit = get_tolerance_by_dtype(dtype)
    
    if npu_output.shape != golden_output.shape:
        raise ValueError(
            f"Shape mismatch: npu {npu_output.shape} vs golden {golden_output.shape}"
        )
    
    ulp_limit = _ULP_FACTOR * _ulp_at_one(dtype)
    max_abs_error_limit = max(fixed_limit, ulp_limit)
    
    # Convert to float64 for computation
    npu_f64 = npu_output.astype(np.float64)
    golden_f64 = golden_output.astype(np.float64)
    
    abs_error = np.abs(npu_f64 - golden_f64)
    element_threshold = atol + rtol * np.abs(golden_f64)
    
    # IEEE special value handling
    finite_golden = np.isfinite(golden_f64)
    finite_npu = np.isfinite(npu_f64)
    
    # Initialize matched array
    element_passed = np.zeros_like(golden_f64, dtype=bool)
    
    # Finite values: use mixed tolerance
    if np.any(finite_golden & finite_npu):
        finite_both = finite_golden & finite_npu
        element_passed[finite_both] = abs_error[finite_both] <= element_threshold[finite_both]
    
    # Special values: NaN/Inf matching per IEEE
    special_golden = ~finite_golden
    special_npu = ~finite_npu
    special_both = special_golden & special_npu
    
    if np.any(special_both):
        # Both are special (NaN or Inf)
        golden_special = golden_f64[special_both]
        npu_special = npu_f64[special_both]
        
        both_nan = np.isnan(golden_special) & np.isnan(npu_special)
        both_pinf = (golden_special == np.inf) & (npu_special == np.inf)
        both_ninf = (golden_special == -np.inf) & (npu_special == -np.inf)
        
        element_passed[special_both] = both_nan | both_pinf | both_ninf
    
    # Mismatched finite/special: not matched (already False)
    
    total = npu_output.size
    if total == 0:
        matched_ratio = 1.0
        max_abs_error = 0.0
    else:
        matched_ratio = float(np.sum(element_passed)) / total
        # max_abs_error only for finite values (special values have inf/nan error)
        finite_mask = finite_golden & finite_npu
        if np.any(finite_mask):
            max_abs_error = float(np.max(abs_error[finite_mask]))
        else:
            max_abs_error = 0.0
    
    ratio_pass = matched_ratio >= required_matched_ratio
    max_err_pass = max_abs_error <= max_abs_error_limit
    is_pass = ratio_pass and max_err_pass
    
    result = {
        'is_pass': is_pass,
        'matched_ratio': matched_ratio,
        'required_matched_ratio': required_matched_ratio,
        'max_abs_error': max_abs_error,
        'max_abs_error_limit': max_abs_error_limit,
        'max_abs_error_limit_fixed': fixed_limit,
        'max_abs_error_limit_ulp': ulp_limit,
        'rtol': rtol,
        'atol': atol,
        'ratio_pass': ratio_pass,
        'max_err_pass': max_err_pass,
        'npu_dtype': str(npu_output.dtype),
        'golden_dtype': str(golden_output.dtype),
        'shape': npu_output.shape,
    }
    
    if not is_pass:
        result['failure_reasons'] = []
        if not ratio_pass:
            result['failure_reasons'].append(
                f'matched_ratio {matched_ratio:.6f} < required {required_matched_ratio}'
            )
        if not max_err_pass:
            result['failure_reasons'].append(
                f'max_abs_error {max_abs_error:.6g} > limit {max_abs_error_limit:.6g}'
            )
    
    return result


def run_precision_check():
    """Run precision check on all test cases."""
    results = []
    all_pass = True
    
    print("=" * 80)
    print("truncate_mod 精度验收测试")
    print("=" * 80)
    print()
    print("参考标准: ops-precision-standard 混合容差 (含 IEEE 特殊值语义)")
    print("  FP16:  rtol=2^-9 (1.95e-3), atol=2^-9 (1.95e-3), matched_ratio>=0.99, max_err<=max(0.1, 32*ULP)")
    print("  BF16:  rtol=2^-6 (1.56e-2), atol=2^-6 (1.56e-2), matched_ratio>=0.99, max_err<=max(1.0, 32*ULP)")
    print("  FP32:  rtol=2^-10 (9.77e-4), atol=2^-16 (1.53e-5), matched_ratio>=0.99, max_err<=max(0.01, 32*ULP)")
    print()
    
    for case_id, dtype, shape, desc in CASES:
        golden_file = os.path.join(DATA_DIR, f"{case_id}_golden.bin")
        output_file = os.path.join(DATA_DIR, f"{case_id}_output.bin")
        
        if not os.path.exists(golden_file) or not os.path.exists(output_file):
            print(f"SKIP {case_id}: file not found")
            continue
        
        # Read data as original dtype
        golden = read_bin_file(golden_file, dtype)
        output = read_bin_file(output_file, dtype)
        
        # Run check with IEEE handling
        result = check_mixed_tolerance_ieee(output, golden)
        
        status = "✅ PASS" if result['is_pass'] else "❌ FAIL"
        if not result['is_pass']:
            all_pass = False
        
        print(f"{status} {case_id:25s} | {desc}")
        print(f"       matched_ratio={result['matched_ratio']:.6f} (req={result['required_matched_ratio']:.2f})  "
              f"max_abs_error={result['max_abs_error']:.6g} (limit={result['max_abs_error_limit']:.6g})  "
              f"rtol={result['rtol']:.2e} atol={result['atol']:.2e}")
        if not result['is_pass']:
            for reason in result.get('failure_reasons', []):
                print(f"       失败原因: {reason}")
        
        # Clean dtype name for display
        if dtype == bfloat16:
            dtype_display = "bfloat16"
        else:
            dtype_display = str(dtype).replace("<class '", "").replace("'>", "").replace("numpy.", "")
        
        result['case_id'] = case_id
        result['description'] = desc
        result['dtype'] = dtype_display
        result['shape'] = shape
        results.append(result)
    
    # Summary
    print()
    print("=" * 80)
    print("精度验收汇总")
    print("=" * 80)
    
    total = len(results)
    pass_count = sum(1 for r in results if r['is_pass'])
    fail_count = total - pass_count
    pass_rate = pass_count / total if total > 0 else 0
    
    print(f"总用例数: {total}")
    print(f"通过数:   {pass_count}")
    print(f"失败数:   {fail_count}")
    print(f"通过率:   {pass_rate:.2%}")
    print()
    
    # Per dtype summary
    print("各 dtype 达标状态表:")
    print("-" * 80)
    print(f"{'dtype':<12} {'用例数':>6} {'通过':>6} {'失败':>6} {'通过率':>8} {'平均 matched_ratio':>18}")
    print("-" * 80)
    
    for dt in ['float16', 'bfloat16', 'float32']:
        dt_results = [r for r in results if dt in r['dtype'].lower() or (dt == 'bfloat16' and 'bfloat' in r['dtype'].lower())]
        if dt_results:
            dt_pass = sum(1 for r in dt_results if r['is_pass'])
            dt_total = len(dt_results)
            dt_rate = dt_pass / dt_total if dt_total > 0 else 0
            avg_ratio = np.mean([r['matched_ratio'] for r in dt_results])
            print(f"{dt:<12} {dt_total:>6} {dt_pass:>6} {dt_total-dt_pass:>6} {dt_rate:>7.2%} {avg_ratio:>18.6f}")
    
    print("-" * 80)
    print(f"{'总计':<12} {total:>6} {pass_count:>6} {fail_count:>6} {pass_rate:>7.2%} {np.mean([r['matched_ratio'] for r in results]):>18.6f}")
    
    # Determine overall status
    overall_status = "✅ 通过" if all_pass else "❌ 失败"
    print()
    print(f"精度验收状态: {overall_status}")
    
    # Write summary.txt
    summary_path = '/mnt/workspace/code/docs/precision/summary.txt'
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        f.write(f"**精度验收状态**: {overall_status}\n\n")
        f.write(f"| dtype | shape | rtol | atol | max_error | 达标状态 |\n")
        f.write(f"|-------|-------|------|------|-----------|----------|\n")
        
        for r in results:
            shape_str = str(r['shape'])
            status_mark = "✅" if r['is_pass'] else "❌"
            f.write(f"| {r['dtype']} | {shape_str} | {r['rtol']:.2e} | {r['atol']:.2e} | {r['max_abs_error']:.6g} | {status_mark} |\n")
        
        f.write(f"\n")
        f.write(f"**关键指标**:\n")
        f.write(f"- 总用例数: {total}\n")
        f.write(f"- 通过: {pass_count}\n")
        f.write(f"- 失败: {fail_count}\n")
        f.write(f"- 通过率: {pass_rate:.2%}\n")
        
        if not all_pass:
            f.write(f"\n**不达标诊断**:\n")
            for r in results:
                if not r['is_pass']:
                    f.write(f"- {r['case_id']} ({r['description']}): ")
                    for reason in r.get('failure_reasons', []):
                        f.write(f"{reason}; ")
                    f.write(f"\n")
                    f.write(f"  问题类型: 精度问题\n")
                    f.write(f"  诊断结论: 需检查算子实现中的数值稳定性、混合精度策略或累积误差处理\n")
    
    print(f"\n报告已写入: {summary_path}")
    return all_pass


if __name__ == '__main__':
    success = run_precision_check()
    sys.exit(0 if success else 1)