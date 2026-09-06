/*!
 * \file relu.cpp
 * \brief Relu 算子 kernel 入口
 */

#include "relu.h"

enum class ReluTilingKey : uint32_t
{
    SMALL_HALF = 0,
    SMALL_FLOAT = 1,
    LARGE_HALF = 2,
    LARGE_FLOAT = 3,
};

template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::SMALL_HALF)) {
        AscendC::InitSocState();
        NsRelu::ReluStatic<half, 1024> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::SMALL_FLOAT)) {
        AscendC::InitSocState();
        NsRelu::ReluStatic<float, 1024> op;
        op.Init(x, y);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::LARGE_HALF)) {
        AscendC::TPipe pipe;
        NsRelu::Relu<half, 11520> op;
        op.Init(x, y, &pipe);
        op.Process();
    } else if constexpr (schMode == static_cast<uint32_t>(ReluTilingKey::LARGE_FLOAT)) {
        AscendC::TPipe pipe;
        NsRelu::Relu<float, 11520> op;
        op.Init(x, y, &pipe);
        op.Process();
    }
}
