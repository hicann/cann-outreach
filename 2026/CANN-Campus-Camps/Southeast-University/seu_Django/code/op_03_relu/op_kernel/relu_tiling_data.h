/*!
 * \file relu_tiling_data.h
 * \brief Shared host/device tiling data; lengths are element counts.
 */
#ifndef RELU_TILING_DATA_H
#define RELU_TILING_DATA_H

#include <cstdint>

struct ReluTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 1;
    int64_t ubFactor = 0;
};

#endif // RELU_TILING_DATA_H
