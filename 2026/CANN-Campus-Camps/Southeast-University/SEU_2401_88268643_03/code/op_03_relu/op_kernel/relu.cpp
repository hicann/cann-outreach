/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

constexpr int RELU_DT_FLOAT = 0;
constexpr int RELU_DT_FLOAT16 = 1;

template <typename T>
__aicore__ inline void RunRelu(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    if (tilingData->totalNum == 0) {
        return;
    }
    if (tilingData->blockFactor <= tilingData->ubFactor) {
        NsRelu::Relu<T, 1> op;
        op.Init(x, y, tilingData);
        op.Process();
    } else {
        NsRelu::Relu<T, 2> op;
        op.Init(x, y, tilingData);
        op.Process();
    }
}

template <int DT_X>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    if constexpr (DT_X == RELU_DT_FLOAT) {
        RunRelu<float>(x, y, &tilingData);
    } else if constexpr (DT_X == RELU_DT_FLOAT16) {
        RunRelu<half>(x, y, &tilingData);
    }
}
