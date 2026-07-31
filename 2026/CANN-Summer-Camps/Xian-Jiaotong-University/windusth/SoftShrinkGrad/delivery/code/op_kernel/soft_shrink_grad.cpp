/*!
 * \file soft_shrink_grad.cpp
 * \brief SoftShrinkGrad 算子 kernel 入口
 */

#include "soft_shrink_grad.h"

enum class SoftShrinkGradTilingKey : uint32_t
{
    TILING_KEY_SOFTSHRINKGRAD_MODE_0 = 0,
    TILING_KEY_SOFTSHRINKGRAD_MODE_1 = 1,
    TILING_KEY_SOFTSHRINKGRAD_MODE_2 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void soft_shrink_grad(GM_ADDR input_grad, GM_ADDR input_x, GM_ADDR output_y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftShrinkGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftShrinkGradTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(SoftShrinkGradTilingKey::TILING_KEY_SOFTSHRINKGRAD_MODE_0)) {
        NsSoftShrinkGrad::SoftShrinkGrad<half> op;
        op.Init(input_grad, input_x, output_y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SoftShrinkGradTilingKey::TILING_KEY_SOFTSHRINKGRAD_MODE_1)) {
        NsSoftShrinkGrad::SoftShrinkGrad<float> op;
        op.Init(input_grad, input_x, output_y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SoftShrinkGradTilingKey::TILING_KEY_SOFTSHRINKGRAD_MODE_2)) {
        NsSoftShrinkGrad::SoftShrinkGrad<bfloat16_t> op;
        op.Init(input_grad, input_x, output_y, &tilingData);
        op.Process();
    }
}
