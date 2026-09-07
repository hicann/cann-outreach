// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;
    uint32_t tileLength; // Elements per tile; its byte size is 32-byte aligned.
};
