/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    int64_t totalLength; // 总元素数量
    int64_t tileNum;     // 单核内分块数（tileLength 已按 BUFFER_NUM 折半，见 relu.h）
};
#endif
