/*!
 * \file relu_tiling_data.h
 * \brief tiling data struct
 */

#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

struct ReluTilingData {
    uint32_t totalLength; // 总元素个数（x/y 一致），本题为 8 * 2048
    uint32_t tileNum;     // 每核内 tile 组数（每组 BUFFER_NUM 块）
    uint32_t dtype;       // 数据类型编码：0=fp32, 1=fp16（host 侧写入，kernel 侧据此分支）
};
#endif
