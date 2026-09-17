/* ------------------------------------------------------------------------ */
/*  Copyright (c), Huawei Technologies Co., Ltd.                            */
/*                                                                          */
/*  Create date: 2026-01-23                                                 */
/*  Revision id: 1 (round3: 补全 tiling 数据结构)                            */
/*  Author: Baidu PaddlePaddle Team                                         */
/*  Function: TanhCustom 算子 Tiling 参数定义（Host 与 Kernel 共享）          */
/* ------------------------------------------------------------------------ */

#ifndef TANH_CUSTOM_TILING_H
#define TANH_CUSTOM_TILING_H

#include <stdint.h>

struct TanhCustomTilingData {
    uint64_t blockDim;  // 参与计算的SIMD核数量
    uint64_t totalSize; // 输入/输出总元素个数
};

#endif // TANH_CUSTOM_TILING_H
