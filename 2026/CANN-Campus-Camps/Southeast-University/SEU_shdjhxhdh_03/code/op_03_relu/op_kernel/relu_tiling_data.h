#ifndef _RELU_TILING_DATA_H_
#define _RELU_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"


struct ReluTilingData {

    // 总元素数量
    uint32_t totalLength;

    // 每个核处理长度
    uint32_t blockLength;

    // tile数量
    uint32_t tileNum;

};


#endif