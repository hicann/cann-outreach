/*!
 * \file truncate_div_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _TRUNCATEDIV_TILING_DATA_H_
#define _TRUNCATEDIV_TILING_DATA_H_

#include <cstdint>

constexpr int64_t kMaxDim = 8;  // 最大维度数

struct TruncateDivTilingData {
    int64_t totalNum = 0;       // 广播后总元素数量
    int64_t blockFactor = 1;    // 每个核处理的元素数量
    int64_t ubFactor = 0;       // 每次 UB 循环处理的元素数量
    int64_t x1ElemNum = 0;      // x1 原始元素数（广播前）
    int64_t x2ElemNum = 0;      // x2 原始元素数（广播前）
    int64_t dimNum = 0;         // 广播后维度数（0=标量）
    int64_t outShape[kMaxDim];  // 广播后输出形状
    int64_t x1Shape[kMaxDim];   // x1 形状（已补1对齐）
    int64_t x2Shape[kMaxDim];   // x2 形状（已补1对齐）
    int64_t tmpSize = 0;        // Trunc 临时缓冲区大小（字节）
};
#endif
