/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */
#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_
struct ReluTilingData {
    int32_t totalLength; // 待处理数据总长度（单位：元素个数），例如 8 * 2048
    int32_t tileNum;     // 单核内将数据进一步切分为多少块（Tile），例如 8
};
#endif