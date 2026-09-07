// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    // All lengths are element counts, shared by Host and Kernel.
    uint32_t length;
    uint32_t blockLength;
    uint32_t tileLength;
};
