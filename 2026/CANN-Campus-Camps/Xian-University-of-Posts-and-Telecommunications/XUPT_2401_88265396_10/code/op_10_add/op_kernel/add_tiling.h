// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct AddTilingData {
    // 输入Tensor总元素数量
    uint32_t totalLength;

    // 每个AIV Core负责的元素数量
    uint32_t blockLength;

    // 每次搬入UB进行计算的元素数量
    uint32_t tileLength;
};