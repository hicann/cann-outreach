/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0
 * Tiling 常量和结构体 - kernel 和 host 共用
 */

#pragma once

#include <cstdint>

// 单次搬运/处理的数据量（half 元素数）
// half(2B) × 8192 = 16KB per buffer, 3 buffers = 48KB UB, 远小于 240KB 上限
constexpr uint32_t TILE_LENGTH = 8192;
constexpr uint32_t DOUBLE_BUFFER = 2;

// Tiling 数据结构 -- 向核函数传递运行时参数
struct InplaceRsqrtTilingData {
    uint32_t blockNum;           // 总 block 数（= used core 数）
    uint64_t totalLength;        // 总元素数
    uint64_t numPerBlock;        // 每个 block 处理的元素数（对齐到 TILE_LENGTH）
    uint64_t tailNumLastBlock;   // 最后一个 block 的尾数
};
