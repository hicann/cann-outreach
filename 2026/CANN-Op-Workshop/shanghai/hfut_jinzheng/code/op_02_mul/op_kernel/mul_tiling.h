// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;   // 待处理元素总数
    uint32_t tileNum;  // 单核内切分的 Tile 数
};
