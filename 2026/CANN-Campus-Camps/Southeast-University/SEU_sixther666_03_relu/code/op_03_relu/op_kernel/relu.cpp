/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

enum class ReluTilingKey : uint32_t
{
    TILING_KEY_FP16_FAST = 0,
    TILING_KEY_FP32_FAST = 1,
    TILING_KEY_FP16_GENERAL = 2,
    TILING_KEY_FP32_GENERAL = 3,
};

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_FP16_FAST)) {
        AscendC::TPipe pipe;
        NsRelu::Relu<half, true> op;
        op.Init(x, y, nullptr, &pipe);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_FP32_FAST)) {
        AscendC::TPipe pipe;
        NsRelu::Relu<float, true> op;
        op.Init(x, y, nullptr, &pipe);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_FP16_GENERAL)) {
        GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
        AscendC::TPipe pipe;
        NsRelu::Relu<half, false> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_FP32_GENERAL)) {
        GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
        AscendC::TPipe pipe;
        NsRelu::Relu<float, false> op;
        op.Init(x, y, &tilingData, &pipe);
        op.Process();
    }
}
