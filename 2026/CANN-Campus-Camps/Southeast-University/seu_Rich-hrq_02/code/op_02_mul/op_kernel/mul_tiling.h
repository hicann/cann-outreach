// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 输入总元素个数
    uint32_t tileNum;      // 每核内的分块数
};

