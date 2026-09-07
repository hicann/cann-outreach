/*!
 * \file relu.cpp
 * \brief Relu kernel entry.
 */
#include "relu.h"

enum class ReluTilingKey : uint32_t {
    TILING_KEY_RELU_MODE_0 = 0,
    TILING_KEY_RELU_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y,
                               GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_RELU_MODE_0)) {
        NsRelu::Relu<half> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_RELU_MODE_1)) {
        NsRelu::Relu<float> op;
        op.Init(x, y, &tilingData);
        op.Process();
    }
}
