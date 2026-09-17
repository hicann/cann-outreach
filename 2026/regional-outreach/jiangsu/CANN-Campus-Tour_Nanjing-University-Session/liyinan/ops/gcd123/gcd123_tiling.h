/**
 * @file gcd123_tiling.h
 * @brief Tiling constants/struct shared by host and kernel for gcd123.
 */
#pragma once

#include <cstdint>

// Elements processed per tile. Keep modest so float/int32 workspace fits in UB.
constexpr uint32_t TILE_LENGTH = 256;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t EUCLID_ITERS = 16;

struct Gcd123TilingData {
    uint32_t blockNum;
    uint64_t totalLength;
    uint64_t numPerCore;
    uint64_t tailNumLastCore;
};
