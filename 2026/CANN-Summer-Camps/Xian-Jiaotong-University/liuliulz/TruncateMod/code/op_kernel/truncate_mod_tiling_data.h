/*!
 * \file truncate_mod_tiling_data.h
 * \brief tiling data struct
 */
#ifndef _TRUNCATEMOD_TILING_DATA_H_
#define _TRUNCATEMOD_TILING_DATA_H_

struct TruncateModTilingData {
    int64_t totalNum = 0;        // 输出总元素数
    int64_t blockFactor = 1;     // 每个核处理的元素数
    int64_t ubFactor = 0;        // 每次 UB 循环处理的元素数
    int64_t x1Total = 0;         // x1 总元素数 (=totalNum 则同 shape，=1 则标量广播)
    int64_t x2Total = 0;         // x2 总元素数
    int64_t x1LastDim = 0;       // x1 最后一维大小（用于广播偏移计算）
    int64_t x2LastDim = 0;       // x2 最后一维大小
    int64_t outLastDim = 0;      // 输出最后一维大小
};
#endif
