/*!
 * \file square_tiling_data.h
 * \brief Square 算子 Tiling 数据定义
 */

#ifndef _SQUARE_TILING_DATA_H_
#define _SQUARE_TILING_DATA_H_

#include <cstdint>

struct SquareTilingData {
    // 任意多维输入张量展平后的总元素数
    int64_t totalNum;

    // 每个 AIV 核最多处理的元素数
    // 按 32 字节对应的元素数对齐
    int64_t blockFactor;

    // 单次搬入 UB 并计算的元素数
    // 按 32 字节对应的元素数对齐
    int64_t ubFactor;
};

#endif // _SQUARE_TILING_DATA_H_