// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t tailNum;
    uint32_t tailLength;
    uint32_t tileLength;
    uint32_t alignNum;
};