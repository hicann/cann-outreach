// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;       // 总元素数
    uint32_t blockLength;  // 每个AI Core处理的元素数
    uint32_t tileLength;   // 每次搬运/计算的元素数
    uint32_t tileNum;      // 每个AI Core需要处理的Tile数量
};
