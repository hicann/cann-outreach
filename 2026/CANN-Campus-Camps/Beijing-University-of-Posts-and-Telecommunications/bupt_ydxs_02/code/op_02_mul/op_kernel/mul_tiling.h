// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    // 输入数据总元素个数
    uint32_t length;

    // 每个Core处理的数据量
    uint32_t blockLength;

    // 每个Tile处理的数据量
    uint32_t tileLength;

    // Tile数量
    uint32_t tileNum;

    // 最后一个Tile的数据量
    uint32_t lastTileLength;
};