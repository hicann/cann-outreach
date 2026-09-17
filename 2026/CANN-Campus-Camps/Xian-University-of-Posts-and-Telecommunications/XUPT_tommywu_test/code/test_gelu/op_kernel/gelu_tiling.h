// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t totalLength;  // 总元素个数
    uint32_t tileNum;      // 每个核上的分块数
};