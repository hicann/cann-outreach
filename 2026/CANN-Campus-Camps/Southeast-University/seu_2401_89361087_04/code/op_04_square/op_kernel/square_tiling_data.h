/*!
 * \file square_tiling_data.h
 * \brief Square tiling data
 */

#ifndef _SQUARE_TILING_DATA_H_
#define _SQUARE_TILING_DATA_H_

#include <cstdint>

struct SquareTilingData {
    // 整个Tensor的元素总数
    int64_t totalNum;

    // 每个AI Core最多负责的元素数量
    int64_t blockFactor;

    // 每次搬入UB处理的最大元素数量
    int64_t ubFactor;
};

#endif
