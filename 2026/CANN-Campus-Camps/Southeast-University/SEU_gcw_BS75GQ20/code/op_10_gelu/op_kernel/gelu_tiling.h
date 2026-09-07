#ifndef GELU_TILING_H
#define GELU_TILING_H

#include <cstdint>

struct GeluTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
};

#endif  // GELU_TILING_H