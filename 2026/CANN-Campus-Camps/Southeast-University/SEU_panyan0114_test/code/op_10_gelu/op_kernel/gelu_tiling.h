// Tiling structure for GELU.
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint64_t totalLength;   // total number of elements in the flattened tensor
    uint64_t blockLength;   // max number of elements processed by one AI Core
    uint32_t tileLength;    // max number of elements processed per UB tile
};
