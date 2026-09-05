#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;   // 总元素数
    uint32_t tileNum;  // 每个核的逻辑 tile 数
};