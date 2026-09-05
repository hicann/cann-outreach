/*!
 * \file relu.cpp
 * \brief Relu leaderboard-optimized kernel entry
 */

#include "relu.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

enum class ReluTilingKey : uint32_t
{
    TILING_KEY_RELU_MODE_0 = 0,
    TILING_KEY_RELU_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    (void)workspace;

    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_RELU_MODE_0)) {
        NsRelu::ReluStaticImpl<half>(x, y, &tilingData);
    }

    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::TILING_KEY_RELU_MODE_1)) {
        NsRelu::ReluStaticImpl<float>(x, y, &tilingData);
    }
}
