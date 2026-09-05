/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    // int64_t totalNum = 0;     // 总元素数量
    // int64_t blockFactor = 1;  // 每个核处理的元素数量
    // int64_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
    uint32_t smallCoreDataNum;  // 小核处理的总数据量（元素个数）
    uint32_t bigCoreDataNum;    // 大核处理的总数据量（元素个数）
    uint32_t finalBigTileNum;   // 大核数据搬运总批次次数
    uint32_t finalSmallTileNum; // 小核数据搬运总批次次数
    uint32_t tileDataNum;       // 单核单次可搬运数据量（元素个数）
    uint32_t smallTailDataNum;  // 小核最后一批处理数据量（元素个数）
    uint32_t bigTailDataNum;    // 大核最后一批处理数据量（元素个数）
    uint32_t tailBlockNum;      // 大核个数（余数分配的核数量）
};
#endif
