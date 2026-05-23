/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// sparse_gemm Tiling 结构体定义
// 用于 Host 和 Kernel 共享 Tiling 参数

#ifndef SPARSE_GEMM_TILING_H
#define SPARSE_GEMM_TILING_H

#include <cstdint>

// Tiling 数据结构体
struct SparseGemmTilingData {
    // 矩阵维度
    uint64_t m;          // A 矩阵的行数 (M)
    uint64_t n;          // B 矩阵的列数 (N)
    uint64_t k;          // A 矩阵的列数 / B 矩阵的行数 (K)
    
    // Tiling 参数
    uint64_t baseM;      // M 维度的基本块大小
    uint64_t baseN;      // N 维度的基本块大小
    uint64_t baseK;      // K 维度的基本块大小
    
    // 多核相关
    uint64_t usedCoreNum;  // 使用的核数
    
    // 尾块处理
    uint64_t mTailCnt;     // M 维度尾块数量
    uint64_t nTailCnt;     // N 维度尾块数量
    uint64_t mBaseTailSplitCnt;  // M 维度基本块尾块分割数
    uint64_t nBaseTailSplitCnt;  // N 维度基本块尾块分割数
    uint64_t mTailMain;    // M 维度主尾块大小
    uint64_t nTailMain;    // N 维度主尾块大小
    
    // 稀疏格式相关
    uint64_t kSparse;      // K 维度稀疏格式大小 (K/2)
    uint64_t kIndex;       // K 维度索引大小 (K/8)
    
    // L1 缓冲区深度
    uint64_t kL1;          // K 维度 L1 缓冲区深度
    
    // L0C 双缓冲
    uint64_t l0cDB;        // L0C 双缓冲标志
    
    // Batch 相关
    uint64_t batch;        // Batch 大小
};

#endif // SPARSE_GEMM_TILING_H
