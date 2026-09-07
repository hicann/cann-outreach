/*!
 * \file square_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _SQUARE_TILING_DATA_H_
#define _SQUARE_TILING_DATA_H_

struct SquareTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // "小核"每个核处理的元素数量（基准块大小）
    int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
    int64_t bigCoreNum = 0;   // "大核"数量：编号在前的 bigCoreNum 个核，每个核比基准块多处理 1 个元素，
                              // 用于把 totalNum % coreNum 的余数均匀摊到前面若干个核上，
                              // 避免余数全部堆在最后一个"尾核"上导致该核成为拖尾、其余核空闲等待
};
#endif