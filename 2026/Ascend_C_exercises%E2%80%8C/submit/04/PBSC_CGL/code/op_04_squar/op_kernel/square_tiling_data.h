/*!
 * \file square_tiling_data.h
 * \brief Square tiling data
 */

#ifndef _SQUARE_TILING_DATA_H_
#define _SQUARE_TILING_DATA_H_

#include <cstdint>

struct SquareTilingData {
    // Tensor 总元素数
    int64_t totalNum;

    // 每个 Core 分配的元素数量
    int64_t blockFactor;

    // 每次 UB 处理的元素数量
    int64_t ubFactor;
};

#endif