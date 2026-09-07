// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength; // 待处理数据总长度（元素个数），8 * 2048 = 16384
    uint32_t tileNum;     // 单核内切分块数
};