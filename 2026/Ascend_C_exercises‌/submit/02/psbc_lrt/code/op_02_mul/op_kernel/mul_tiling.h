// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;      // 总元素个数
    uint32_t tileNum;          // 分块数量
    uint32_t tileLength;       // 每块元素个数（已对齐，最后一块除外）
    uint32_t lastTileLength;   // 最后一块元素个数
};