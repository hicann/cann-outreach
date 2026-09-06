// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    // 整个输入Tensor的元素数量
    uint32_t totalLength;

    // 单个AIV Core最多处理的元素数量
    uint32_t blockLength;

    // 单次搬入UB并进行Mul计算的元素数量
    uint32_t tileLength;
};