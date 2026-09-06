// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength; // total element count, e.g. 8 * 2048
    uint32_t tileNum;     // tiles per core
};
