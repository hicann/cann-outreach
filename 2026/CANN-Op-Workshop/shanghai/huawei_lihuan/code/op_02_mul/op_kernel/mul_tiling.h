// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength; // 待处理数据总长度（单位：元素个数），例如 8 * 2048
    uint32_t tileNum;     // 单核内将数据进一步切分为多少块（Tile），例如 8
};