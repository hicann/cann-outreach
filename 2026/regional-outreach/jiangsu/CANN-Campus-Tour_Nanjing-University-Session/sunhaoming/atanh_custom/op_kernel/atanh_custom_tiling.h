/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATANH_CUSTOM_TILING_H
#define ATANH_CUSTOM_TILING_H

// TILE_LENGTH: 每次处理的数据量（元素数），受限于 UB 大小和临时 buffer 需求
// Atanh half 类型需要 4× 额外 tmpBuffer (float32)，所以 TILE_LENGTH 不宜过大
// UB 约 256KB，float32 4 字节，4× = 16 字节/元素
// 256KB / (2 + 4×4) = 256KB / 18B ≈ 14563，取 8192 留余量
constexpr uint32_t TILE_LENGTH = 8192;

struct AtanhTilingData {
    uint32_t totalLength;     // 总元素数
    uint32_t blockNum;        // 使用的核数
    uint32_t numPerCore;      // 每核分配的元素数（向上对齐 TILE_LENGTH）
    uint32_t tailNumLastCore; // 最后一核的实际元素数
};

#endif
