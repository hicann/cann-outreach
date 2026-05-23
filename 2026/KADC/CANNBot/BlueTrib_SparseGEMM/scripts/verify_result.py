#!/usr/bin/env python3
"""
sparse_gemm 结果验证脚本
验证算子输出是否符合预期
"""

import numpy as np
import os
import argparse
import sys

def load_shape_info(input_dir):
    """加载形状信息"""
    shape_file = os.path.join(input_dir, 'shape.txt')
    shape_info = {}
    
    with open(shape_file, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
                shape_info[key] = value
    
    return shape_info

def verify_result(output_file, golden_file, shape_info, atol=1e-6, rtol=1e-6):
    """验证结果是否匹配"""
    # 加载数据
    c_shape_str = shape_info['c_shape'].strip('()')
    c_shape = tuple(map(int, c_shape_str.split(',')))
    
    output = np.fromfile(output_file, dtype=np.float32).reshape(c_shape)
    golden = np.fromfile(golden_file, dtype=np.float32).reshape(c_shape)
    
    # 计算差异
    diff = np.abs(output - golden)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    # 检查是否满足精度要求
    is_close = np.allclose(output, golden, atol=atol, rtol=rtol)
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'is_close': is_close,
        'output_shape': output.shape,
        'golden_shape': golden.shape
    }

def main():
    parser = argparse.ArgumentParser(description='Verify sparse_gemm result')
    parser.add_argument('--output', type=str, default='./output/output.bin', help='Output file path')
    parser.add_argument('--golden', type=str, default='./input/golden.bin', help='Golden file path')
    parser.add_argument('--input_dir', type=str, default='./input', help='Input directory with shape info')
    parser.add_argument('--atol', type=float, default=1e-6, help='Absolute tolerance')
    parser.add_argument('--rtol', type=float, default=1e-6, help='Relative tolerance')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.output):
        print(f"Error: Output file not found: {args.output}")
        return 1
    
    if not os.path.exists(args.golden):
        print(f"Error: Golden file not found: {args.golden}")
        return 1
    
    if not os.path.exists(os.path.join(args.input_dir, 'shape.txt')):
        print(f"Error: Shape file not found: {args.input_dir}/shape.txt")
        return 1
    
    # 加载形状信息
    shape_info = load_shape_info(args.input_dir)
    
    # 验证结果
    print("Verifying sparse_gemm result...")
    print(f"  Output: {args.output}")
    print(f"  Golden: {args.golden}")
    print(f"  Tolerance: atol={args.atol}, rtol={args.rtol}")
    
    result = verify_result(args.output, args.golden, shape_info, args.atol, args.rtol)
    
    # 输出结果
    print(f"\nResults:")
    print(f"  Output shape: {result['output_shape']}")
    print(f"  Golden shape: {result['golden_shape']}")
    print(f"  Max diff: {result['max_diff']:.6e}")
    print(f"  Mean diff: {result['mean_diff']:.6e}")
    print(f"  Is close: {result['is_close']}")
    
    if result['is_close']:
        print("\n✅ Verification PASSED!")
        return 0
    else:
        print("\n❌ Verification FAILED!")
        
        if args.verbose:
            # 输出详细差异信息
            output = np.fromfile(args.output, dtype=np.float32)
            golden = np.fromfile(args.golden, dtype=np.float32)
            
            # 找到最大差异的位置
            diff = np.abs(output - golden)
            max_idx = np.argmax(diff)
            
            print(f"\nDetailed diff info:")
            print(f"  Max diff at index {max_idx}:")
            print(f"    Output: {output[max_idx]}")
            print(f"    Golden: {golden[max_idx]}")
            print(f"    Diff: {diff[max_idx]}")
            
            # 统计差异分布
            print(f"\nDiff distribution:")
            print(f"  diff < 1e-6: {np.sum(diff < 1e-6)} elements")
            print(f"  diff < 1e-4: {np.sum(diff < 1e-4)} elements")
            print(f"  diff < 1e-2: {np.sum(diff < 1e-2)} elements")
            print(f"  diff >= 1e-2: {np.sum(diff >= 1e-2)} elements")
        
        return 1

if __name__ == '__main__':
    sys.exit(main())
