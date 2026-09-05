// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;      // 总元素个数
    uint32_t blockLength; // 单核处理的元素个数(32字节对齐)
    uint32_t tileLength;  // 单次循环(tile)处理的元素个数
    uint32_t tileNum;     // 单核循环次数
};