// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    // 输入Tensor总元素个数
    uint32_t length;

    // 每个AI Vector Core处理的元素个数
    uint32_t blockLength;

    // 每个Core上的逻辑tile数量
    uint32_t tileNum;

    // 每次CopyIn / Compute / CopyOut处理的元素数
    uint32_t tileLength;
};
