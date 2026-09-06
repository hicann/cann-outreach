/*!
 * \file relu_tiling_data.h
 * \brief Relu tiling data definition
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

#include <cstdint>

struct ReluTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 1;
    int64_t ubFactor = 0;
};

#endif
