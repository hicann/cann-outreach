/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

// 核函数入口：使用标准的纯数字常量进行 TILING_KEY 匹配
extern "C" __global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);

    // 严禁在此处使用 static_cast，必须直接写纯数字字面量 0 和 1
    if (TILING_KEY_IS(0)) {
        // Mode 0: float16 (half)
        NsRelu::Relu<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        // Mode 1: float32 (float)
        NsRelu::Relu<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}