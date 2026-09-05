/*!
 * \file square_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SQUARE_TILING_DATA_H_
#define _SQUARE_TILING_DATA_H_

struct SquareTilingData {
    int64_t totalNum = 0;     // 输入张量展平后的总元素数
    int64_t blockFactor = 0;  // 单个 AIV 核最多处理的元素数
    int64_t ubFactor = 0;     // 单次 UB 搬运与计算的元素数（32 字节对齐）
};
#endif
