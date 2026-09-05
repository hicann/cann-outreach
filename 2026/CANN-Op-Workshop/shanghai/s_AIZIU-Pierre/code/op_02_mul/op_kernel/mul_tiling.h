// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
     uint32_t totalLength;   // 全部元素个数
    uint32_t tileNum;       // 每个核内部再切成几块
};