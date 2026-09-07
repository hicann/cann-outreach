/*!
 * \file square_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SQUARE_TILING_DATA_H_
#define _SQUARE_TILING_DATA_H_

#include <cstdint>

struct SquareTilingData {
    uint64_t totalNum;     // 输入张量的总元素数
    uint64_t blockFactor;  // 每个 AI Vector Core 最多处理的元素数
    uint64_t ubFactor;     // 单次 GM <-> UB 搬运/计算的元素数（32B 对齐）
};

#endif