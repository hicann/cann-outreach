/*!
 * \file truncate_mod_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _TRUNCATEMOD_TILING_DATA_H_
#define _TRUNCATEMOD_TILING_DATA_H_

#include <cstdint>

struct TruncateModTilingData {
    int64_t totalNum = 0;
    int64_t blockFactor = 0;
    int64_t ubFactor = 0;
    int64_t outDims = 0;
    int64_t x1Dims = 0;
    int64_t x2Dims = 0;
    int64_t outShape[8] = {0};
    int64_t x1Shape[8] = {0};
    int64_t x2Shape[8] = {0};
};
#endif
