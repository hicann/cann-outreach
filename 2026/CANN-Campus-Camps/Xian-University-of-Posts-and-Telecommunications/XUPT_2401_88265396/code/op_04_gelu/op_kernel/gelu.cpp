/*!
 * \file gelu.cpp
 * \brief Gelu 算子 kernel 入口
 */

#include "gelu.h"

enum class GeluTilingKey : uint32_t
{
    TILING_KEY_GELU_MODE_0 = 0,
    TILING_KEY_GELU_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(GeluTilingKey::TILING_KEY_GELU_MODE_0)) {
        NsGelu::Gelu<half> op;
        op.Init(input_x, output, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(GeluTilingKey::TILING_KEY_GELU_MODE_1)) {
        NsGelu::Gelu<float> op;
        op.Init(input_x, output, &tilingData);
        op.Process();
    }
}
