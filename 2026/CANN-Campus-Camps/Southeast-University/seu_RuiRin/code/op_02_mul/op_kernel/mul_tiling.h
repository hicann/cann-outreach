// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;       // 总元素数量
    uint32_t blockLength;  // 每个核处理的元素数量
};
