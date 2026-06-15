/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0
 *
 * Tiling 常量 + 结构体 — Sign 算子
 * kernel 和 host 共用
 */

#pragma once

#include <cstdint>

// TILE_LENGTH: 单次搬运/处理的元素数 (half)
// 约束:
//   1. UB 用量: 6 × TILE_LENGTH × sizeof(half) + 2 × TILE_LENGTH × sizeof(uint8_t)
//              = (12 + 2) × TILE_LENGTH = 14 × 4096 = 57344 bytes, 远小于 240KB UB
//   2. Compare 256B 对齐: half 下 TILE_LENGTH 需为 128 倍数
//   3. TILE_LENGTH = 4096 = 32 × 128 ✓
constexpr uint32_t TILE_LENGTH = 4096;
constexpr uint32_t DOUBLE_BUFFER = 2;

struct SignTilingData {
    uint32_t blockNum;           // 总 block 数
    uint64_t totalLength;        // 总元素数
    uint64_t numPerBlock;        // 每个 block 处理的元素数
    uint64_t tailNumLastBlock;   // 最后一个 block 的尾数
};
