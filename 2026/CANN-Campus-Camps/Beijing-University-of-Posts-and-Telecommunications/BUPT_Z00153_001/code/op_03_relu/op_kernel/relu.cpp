/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口（非模板核函数，dtype 经 tiling 传递后内核内分支）
 */

#include "relu.h"

__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    // dtype: 0=fp32, 1=fp16（host 侧 ReluTilingFunc 写入）
    if (tilingData.dtype == 1) {
        // fp16: tileNum=1 -> 单块 2048 元素（4KB），1 次 DMA 大 burst 压固定开销
        NsRelu::Relu<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else {
        // fp32: tileNum=2 -> 2 轮 x 1024 双缓冲，流水掩盖搬运（PASS 版等价配置）
        NsRelu::Relu<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
