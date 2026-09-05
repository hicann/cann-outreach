// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;     // 输入元素总个数
    uint32_t tileNum;         // 分片个数（= 启动的核数）
    uint32_t tileLength;      // 每个分片的元素个数（向上取整）
    uint32_t lastTileLength;  // 最后一个分片的元素个数
};