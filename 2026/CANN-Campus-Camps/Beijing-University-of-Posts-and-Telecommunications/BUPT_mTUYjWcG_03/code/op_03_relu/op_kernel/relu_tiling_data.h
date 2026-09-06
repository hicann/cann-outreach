/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

#include <cstdint>

struct ReluTilingData {
    uint32_t totalNum;
    uint32_t blockFactor;
    uint32_t ubFactor;
};

#endif
