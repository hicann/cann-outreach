#ifndef _SQUARE_TILING_DATA_H_
#define _SQUARE_TILING_DATA_H_

struct SquareTilingData {
    int64_t totalNum = 0;      // 输入 Tensor 总元素数量
    int64_t blockFactor = 0;   // 每个 Core 正常处理的元素数量
    int64_t ubFactor = 0;      // 每个 UB Tile 处理的元素数量
    int64_t tailNum = 0;       // 最后一个 Core 实际处理的元素数量
};

#endif