// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    // 输入Tensor的总元素数量
    uint32_t totalLength;

    // 每个Core内部的分块数量
    uint32_t tileNum;
};