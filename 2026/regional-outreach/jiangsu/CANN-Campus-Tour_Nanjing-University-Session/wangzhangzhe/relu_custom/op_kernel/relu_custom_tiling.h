/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// Tiling 常量和结构体 - ReLU 算子（float16, 4D ND）
// ============================================================================
// 此文件只含纯 C/C++ 语法，不含 __aicore__、__gm__ 等 ASC 关键字
//
// 数学公式: relu(x) = max(0, x)
// 数据类型: float16（逐元素比较，无需精度提升）
// 支持 Shape: 任意 4D ND（如 [N4,N3,N2,N1]）
// ============================================================================

#pragma once

#include <cstdint>

// float16 单元素 2 字节，TILE_LENGTH=8192 对应 UB 单 Buffer 16KB
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t ELEM_BYTE = 2;  // sizeof(half) = 2

// Tiling 数据结构
struct ReluTilingData {
    uint32_t blockNum;          // 使用的 AI Core 数量
    uint64_t totalLength;       // 总元素数（N1*N2*N3*N4）
    uint64_t numPerCore;        // 每个 Core 处理的元素数
    uint64_t tailNumLastCore;   // 最后一个 Core 实际处理的元素数
};
