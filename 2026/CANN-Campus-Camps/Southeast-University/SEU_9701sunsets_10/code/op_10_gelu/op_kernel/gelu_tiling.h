#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t length;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t mode;  // 0: erf exact, 1: polynomial
};