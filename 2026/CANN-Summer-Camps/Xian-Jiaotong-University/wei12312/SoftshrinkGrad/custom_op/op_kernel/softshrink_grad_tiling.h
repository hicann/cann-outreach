#ifndef SOFTSHRINK_GRAD_TILING_H
#define SOFTSHRINK_GRAD_TILING_H

#include <cstdint>

struct SoftshrinkGradTilingData {
    uint64_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;
    float lambd;
};

#endif
