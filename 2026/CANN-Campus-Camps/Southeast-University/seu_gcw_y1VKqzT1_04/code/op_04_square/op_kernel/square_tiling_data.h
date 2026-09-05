/*!
 * \file square_tiling_data.h
 * \brief tiling data struct
 */

#ifndef ASCENDC_LESSON04_SQUARE_TILING_DATA_H
#define ASCENDC_LESSON04_SQUARE_TILING_DATA_H

#include <cstdint>

struct SquareTilingData {
    int64_t totalNum;     // 输入张量总元素数
    int64_t blockFactor;  // 每个核最多处理的元素数
    int64_t ubFactor;     // 单次 UB tile 处理的元素数
};

#endif // ASCENDC_LESSON04_SQUARE_TILING_DATA_H
