// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 元素总数 = 各维乘积（x->GetShapeSize()）
    uint32_t blockNum;     // 逻辑block数 = 启动核数（= SetBlockDim 值）
    uint32_t blockLength;  // 每block基础元素数（512元素对齐）
    uint32_t tileLength;   // 每UB tile元素数（256B对齐）
};