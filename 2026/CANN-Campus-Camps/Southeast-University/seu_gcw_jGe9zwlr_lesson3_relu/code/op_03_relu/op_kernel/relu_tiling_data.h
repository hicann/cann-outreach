/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef RELU_TILING_DATA_H
#define RELU_TILING_DATA_H

#include <cstdint>

struct ReluTilingData {

    // 输入 Tensor 总元素数量
    uint32_t totalNum;

    // 每个 Core 处理的元素数量
    uint32_t blockFactor;

    // 每次 UB 处理的元素数量
    uint32_t ubFactor;

};

#endif // RELU_TILING_DATA_H