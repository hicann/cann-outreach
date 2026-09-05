// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t length;      // 总元素个数
    uint32_t blockFactor; // 每个核处理的元素个数
    uint32_t ubFactor;    // 单次 UB 搬运/计算的元素个数
};

