#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    int64_t totalNum = 0;     // 总元素数量
    int64_t blockFactor = 1;  // 每个tile处理的元素数量
    int64_t ubFactor = 0;     // 未使用
};

#endif