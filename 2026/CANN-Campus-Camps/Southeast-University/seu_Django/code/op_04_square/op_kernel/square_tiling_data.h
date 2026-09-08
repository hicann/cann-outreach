/*!
 * \file square_tiling_data.h
 * \brief Shared host/device tiling data; lengths are element counts.
 */
#ifndef SQUARE_TILING_DATA_H
#define SQUARE_TILING_DATA_H

#include <cstdint>

struct SquareTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 1;
    int64_t ubFactor = 0;
};

#endif // SQUARE_TILING_DATA_H
