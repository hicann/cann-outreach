// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

#ifndef RELU_CUSTOM_TILING_H
#define RELU_CUSTOM_TILING_H

#include "tiling/tiling_api.h"

// Tiling 结构体：Kernel 侧与 Host 侧共用
struct ReluCustomTilingData {
    int32_t totalElements;  // 总元素数（整个 tensor）
    int32_t tileLen;        // 每次处理元素数（对齐到 32B）
    uint32_t usedCoreNum;   // 使用的核数
};

// ReLU 只需要一对 I/O buffer，每元素 4 字节 (2 for in + 2 for out)
constexpr int32_t UB_BYTES_PER_ELEM = 2 * static_cast<int32_t>(sizeof(half));  // = 4

// Host 侧 Tiling 计算
inline void ComputeReluCustomTiling(ReluCustomTilingData &tiling, int32_t totalElements,
                                     int64_t availableCoreNum, int32_t ubAvailableSize)
{
    // 核数 = min(元素数, 可用核数)
    uint32_t usedNumBlocks = (totalElements < availableCoreNum) ?
                              (uint32_t)totalElements : (uint32_t)availableCoreNum;
    usedNumBlocks = (usedNumBlocks == 0) ? 1 : usedNumBlocks;

    // 每核处理元素数（向上取整）
    int32_t perCoreElements = (totalElements + usedNumBlocks - 1) / usedNumBlocks;

    // 计算 tileLen: UB 可用容量 / 每元素所需字节（双缓冲）
    int32_t maxTileByUb = ubAvailableSize / UB_BYTES_PER_ELEM;

    // 32B 对齐（16 个 half）
    constexpr int32_t ALIGN_UNIT = 16;  // 32B / sizeof(half)
    int32_t tileLen = (perCoreElements < maxTileByUb) ? perCoreElements : maxTileByUb;
    tileLen = (tileLen + ALIGN_UNIT - 1) / ALIGN_UNIT * ALIGN_UNIT;

    // 写入 tiling
    tiling.totalElements = totalElements;
    tiling.tileLen = tileLen;
    tiling.usedCoreNum = usedNumBlocks;
}

#endif  // RELU_CUSTOM_TILING_H
