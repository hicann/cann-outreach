/*!
 * \file relu.cpp
 * \brief Relu Kernel入口
 */

#include "relu.h"

enum class ReluTilingKey : uint32_t
{
    TILING_KEY_RELU_MODE_0 = 0,
    TILING_KEY_RELU_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void relu(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(
        ReluTilingData,
        tilingData,
        tiling);

    if constexpr (
        schMode ==
        static_cast<uint32_t>(
            ReluTilingKey::
                TILING_KEY_RELU_MODE_0)) {
        // float16使用显式mask/repeat。
        NsRelu::Relu<half, true> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }

    if constexpr (
        schMode ==
        static_cast<uint32_t>(
            ReluTilingKey::
                TILING_KEY_RELU_MODE_1)) {
        // float32使用count接口。
        NsRelu::Relu<float, false> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}