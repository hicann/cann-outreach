/*!
 * \file truncate_mod_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _TRUNCATEMOD_TILING_DATA_H_
#define _TRUNCATEMOD_TILING_DATA_H_

struct TruncateModTilingData {
    int64_t totalNum = 0;    // 广播后输出总元素数
    int64_t blockFactor = 1; // 每核元素数（512 元素对齐）
    int64_t ubFactor = 0;    // 每次 UB 循环元素数（256B 对齐）
    int64_t x1InnerLen = 0;  // x1 连续运行长度（元素）
    int64_t x1RowCount = 0;  // x1 源行数（= x1Total / x1InnerLen）
    int64_t x2InnerLen = 0;  // x2 连续运行长度（元素）
    int64_t x2RowCount = 0;  // x2 源行数（= x2Total / x2InnerLen）
};
#endif
