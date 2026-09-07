#pragma once

#include <cstdint>

struct GeluTilingData {
    uint64_t totalLength;
    uint64_t largeBlockLength;
    uint64_t smallBlockLength;
    uint32_t largeCoreCount;
    uint32_t tileLength;
};
