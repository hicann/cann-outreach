/*!
 * \file selu_grad.cpp
 * \brief SeluGrad 算子 kernel 入口
 */

#include "selu_grad.h"

enum class SeluGradTilingKey : uint32_t
{
    TILING_KEY_SELUGRAD_MODE_0 = 0,
    TILING_KEY_SELUGRAD_MODE_1 = 1,
    TILING_KEY_SELUGRAD_MODE_2 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void selu_grad(GM_ADDR gradients, GM_ADDR outputs, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SeluGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SeluGradTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SeluGradTilingKey::TILING_KEY_SELUGRAD_MODE_0)) {
        NsSeluGrad::SeluGrad<half> op;
        op.Init(gradients, outputs, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SeluGradTilingKey::TILING_KEY_SELUGRAD_MODE_1)) {
        NsSeluGrad::SeluGrad<float> op;
        op.Init(gradients, outputs, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SeluGradTilingKey::TILING_KEY_SELUGRAD_MODE_2)) {
        NsSeluGrad::SeluGrad<bfloat16_t> op;
        op.Init(gradients, outputs, y, &tilingData);
        op.Process();
    }
}
