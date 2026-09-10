/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    uint64_t totalLength;   // 输入元素总数
    uint64_t blockFormer;   // 每核基础处理量（元素数，512 对齐）
    uint64_t blockTail;     // 尾核实际处理量
    uint64_t blockNum;      // 实际使用的核数
    uint64_t ubFormer;      // 每次 UB 搬入搬出的元素数（256B 对齐）
    uint64_t ubLoop;        // 尾核之外的 UB 循环次数
    uint64_t ubTail;        // 每核最后一次 UB 循环处理的元素数
};
#endif
