// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint64_t length;
    uint64_t blockLength;
    uint32_t tileLength;
    uint32_t erfTmpBytes;
};
