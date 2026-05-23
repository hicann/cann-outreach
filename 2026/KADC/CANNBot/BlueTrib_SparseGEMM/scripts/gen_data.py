#!/usr/bin/env python3
"""
sparse_gemm 测试数据生成脚本
生成 2:4 稀疏格式的测试数据

参考：asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/scripts/gen_data.py

[FIX] 审查报告修复项：
1. 正确处理稀疏格式的 golden 计算
2. 根据索引矩阵恢复完整的 B 矩阵
3. 正确生成 NZ 格式的数据
"""

import numpy as np
import os
import argparse


def densify_and_generate_index(B):
    """稠密化稀疏矩阵B，并生成索引矩阵"""
    N, K = B.shape
    dense_B = np.zeros((N, K // 2), dtype=B.dtype)  # 稠密化后的矩阵
    index_matrix = np.zeros((N, K // 2), dtype=np.uint8)  # 索引矩阵
    index_mask_matrix = np.zeros((N, K // 2), dtype=np.uint32)  # index mask矩阵

    for row in range(N):
        dense_row = []
        index_row = []
        index_mask_row = []
        
        for i in range(0, K, 4):
            block = B[row, i:i+4]
            nonzero_positions = [j for j in range(4) if block[j] != 0]

            # 记录第1和第2个非零元素的索引
            if len(nonzero_positions) == 0:
                index_1 = 0
                index_2 = 0
                index_mask_row.extend([i, i])
            elif len(nonzero_positions) == 1:
                index_1 = nonzero_positions[0] if nonzero_positions[0] < 3 else 0
                index_2 = 0 if nonzero_positions[0] < 3 else 2
                index_mask_row.extend([nonzero_positions[0] + i, i])
            else:
                index_1 = nonzero_positions[0]
                index_2 = nonzero_positions[1] - 1
                index_mask_row.extend([nonzero_positions[0] + i, nonzero_positions[1] + i])
            
            # 记录稠密化后的块
            dense_block = [block[pos] for pos in nonzero_positions[:2]]
            if len(dense_block) < 2:
                dense_block += [0] * (2 - len(dense_block))
            dense_row.extend(dense_block)
            
            # 记录索引
            index_row.extend([index_1, index_2])
        dense_B[row, :] = dense_row
        index_matrix[row, :] = index_row
        index_mask_matrix[row, :] = index_mask_row
    
    return dense_B, index_matrix, index_mask_matrix


def construct_sparse_matrix_B(shape):
    """生成一个指定形状的稀疏矩阵B，每行的每4个元素块至少包含2个零"""
    N, K = shape
    B = np.zeros((N, K), dtype=np.int8)  # 初始化矩阵B为全零
    
    for row in range(N):
        for i in range(0, K, 4):
            block = np.zeros(4, dtype=np.int8)    
            # 随机选择2个位置放置非零元素
            non_zero_positions = np.random.choice(4, 2, replace=False)
            block[non_zero_positions[0]] = np.random.randint(1, 10, dtype=np.int8)
            block[non_zero_positions[1]] = np.random.randint(1, 10, dtype=np.int8)
            # 放置到矩阵B的当前行
            B[row, i:i+4] = block 
    return B


def gen_sparse_golden(A, dense_B, index_mask_matrix):
    """计算稀疏矩阵乘法的 Golden 结果"""
    # 使用 float32 计算，与硬件输出一致
    result_type = np.float32
    M = A.shape[0]
    N = dense_B.shape[0]
    C = np.zeros((M, N), dtype=result_type)
    # 遍历 b 和 index 的每一行
    for r in range(N):
        # 从 a 中根据 index 的第 r 行提取数据
        selected_columns = index_mask_matrix[r]  # 第 r 行的索引
        a_selected = A[:, selected_columns]  # 提取对应列
        
        # 当前 b 第 r 行与提取后的 a_selected 计算矩阵乘法
        # 先计算 int32 结果，再转换为 float32（与硬件行为一致）
        C[:, r] = np.dot(a_selected.astype(np.int32), dense_B[r].astype(np.int32)).astype(result_type)
    return C


def gen_uint2_zn_idx(index_matrix):
    """将索引矩阵转换为 NZ 格式"""
    # K1 = K // 2
    N, K1 = index_matrix.shape
    K2 = K1 // 4
    index = np.zeros((N, K2), dtype=np.uint8)
    for row in range(N):
        index_bytes = []
        index_row = index_matrix[row]
        # 4个uint2 拼成一个unint8
        for j in range(0, len(index_row), 4):
            indices = index_row[j : j + 4]
            uint8_value = sum((idx << (2 * k)) for k, idx in enumerate(indices))
            index_bytes.append(uint8_value)
        
        index[row, :] = index_bytes

    # nd->nz 等价 dn->zn
    ceil_N = int(np.ceil(N / 16) * 16)
    pad_N = int(ceil_N - N)
    ceil_K = int(np.ceil(K2 / 8) * 8)
    pad_K = int(ceil_K - K2)
    index_matrix_nz = np.zeros((ceil_N, ceil_K), dtype=np.uint8)
    index_matrix_nz[:N, :K2] = index

    nz_shape = (ceil_N // 16, 16, ceil_K // 8, 8)
    index_matrix_nz = index_matrix_nz.reshape(nz_shape)
    index_matrix_nz = index_matrix_nz.transpose(2, 0, 1, 3)
    return index_matrix_nz


def main():
    parser = argparse.ArgumentParser(description='Generate test data for sparse_gemm')
    parser.add_argument('--batch', type=int, default=2, help='Batch size')
    parser.add_argument('--m', type=int, default=128, help='M dimension')
    parser.add_argument('--n', type=int, default=128, help='N dimension')
    parser.add_argument('--k', type=int, default=128, help='K dimension (D_H)')
    parser.add_argument('--output_dir', type=str, default='./input', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    batch = args.batch
    m = args.m
    n = args.n
    k = args.k
    
    print(f"Generating sparse_gemm test data:")
    print(f"  batch={batch}, m={m}, n={n}, k={k}")
    print(f"  A shape: [{batch}, {m}, {k}]")
    print(f"  B shape: [{batch}, {n}, {k}] (sparse, will be densified to [{batch}, {n}, {k//2}])")
    print(f"  Index shape: [{batch}, {n}, {k//8}]")
    print(f"  C shape: [{batch}, {m}, {n}]")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    
    c0Size = 32
    
    # 生成每个 batch 的数据
    all_a = []
    all_dense_b = []
    all_idx = []
    all_golden = []
    
    for b in range(batch):
        # 生成 A 矩阵（稠密）
        A_gm = np.random.randint(1, 10, [m, k]).astype(np.int8)
        
        # 构造稀疏 B 矩阵，确保每 4 个元素中至少包含 2 个零
        B_gm = construct_sparse_matrix_B((n, k)).astype(np.int8)
        
        # 4选2稠密化 B 矩阵，并生成对应的 uint8 类型的索引矩阵
        dense_B, index_matrix, index_mask_matrix = densify_and_generate_index(B_gm)
        
        # 生成稀疏矩阵乘 golden
        golden = gen_sparse_golden(A_gm, dense_B, index_mask_matrix)
        
        # 将 uint8 类型的索引矩阵转换成 uint2 类型的索引矩阵，并将其转换成 Zn
        idx_gm = gen_uint2_zn_idx(index_matrix)
        
        # A 矩阵转换为 NZ 格式
        x1_gm = A_gm.reshape((int(m / 16), 16, int(k / c0Size), c0Size))\
            .transpose(2, 0, 1, 3).astype(np.int8)
        
        # B 矩阵（稠密化后）转换为 NZ 格式
        x2_gm = dense_B.reshape((int(n / 16), 16, int(k / 2 / c0Size), c0Size))\
            .transpose(2, 0, 1, 3).astype(np.int8)
        
        all_a.append(x1_gm)
        all_dense_b.append(x2_gm)
        all_idx.append(idx_gm)
        all_golden.append(golden)
    
    # 合并所有 batch 的数据
    a_nz = np.stack(all_a, axis=0)
    dense_b_nz = np.stack(all_dense_b, axis=0)
    idx_nz = np.stack(all_idx, axis=0)
    golden_all = np.stack(all_golden, axis=0)
    
    # 保存数据
    print("Saving data...")
    a_nz.tofile(os.path.join(args.output_dir, 'x1_gm.bin'))
    dense_b_nz.tofile(os.path.join(args.output_dir, 'x2_gm.bin'))
    idx_nz.tofile(os.path.join(args.output_dir, 'idx_gm.bin'))
    golden_all.tofile(os.path.join(args.output_dir, 'golden.bin'))
    
    # 保存形状信息
    with open(os.path.join(args.output_dir, 'shape.txt'), 'w') as f:
        f.write(f"batch={batch}\n")
        f.write(f"m={m}\n")
        f.write(f"n={n}\n")
        f.write(f"k={k}\n")
        f.write(f"a_shape={a_nz.shape}\n")
        f.write(f"b_shape={dense_b_nz.shape}\n")
        f.write(f"idx_shape={idx_nz.shape}\n")
        f.write(f"c_shape={golden_all.shape}\n")
    
    print("Done!")
    print(f"Files saved to {args.output_dir}/")
    print(f"A shape (NZ): {a_nz.shape}")
    print(f"B shape (NZ): {dense_b_nz.shape}")
    print(f"Index shape (NZ): {idx_nz.shape}")
    print(f"Golden shape: {golden_all.shape}")


if __name__ == '__main__':
    main()
