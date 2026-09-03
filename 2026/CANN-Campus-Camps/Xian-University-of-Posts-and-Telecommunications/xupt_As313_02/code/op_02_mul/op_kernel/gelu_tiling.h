
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t length;            // 输入总元素个数
    uint32_t smallCoreDataNum;  // 小核(处理小块数据的核)负责的元素个数
    uint32_t bigCoreDataNum;    // 大核(处理大块数据的核)负责的元素个数
    uint32_t finalSmallTileNum; // 小核需要处理的tile个数
    uint32_t finalBigTileNum;   // 大核需要处理的tile个数
    uint32_t tileDataNum;       // 每个tile的元素个数
    uint32_t smallTailDataNum;  // 小核最后一片tile的元素个数
    uint32_t bigTailDataNum;    // 大核最后一片tile的元素个数
    uint32_t tailBlockNum;      // 处理大块数据的核(大核)个数
};