#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;
    uint32_t blockNum;
    uint32_t blockLen;
    uint32_t tileLen;
};
