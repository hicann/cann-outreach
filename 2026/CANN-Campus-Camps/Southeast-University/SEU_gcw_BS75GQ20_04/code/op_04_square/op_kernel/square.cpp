/*!
 * \file square.cpp
 * \brief Square 算子 kernel 入口
 */

#include "square.h"

enum class SquareTilingKey : uint32_t
{
    // FP16
    TILING_KEY_SQUARE_MODE_0 = 0,

    // FP32
    TILING_KEY_SQUARE_MODE_1 = 1,
};

template <uint32_t schMode>
__global__ __aicore__ void square(
    GM_ADDR input_x,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SquareTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        SquareTilingData,
        tilingData,
        tiling);

    // TPipe放在Kernel入口，不作为Square类成员
    AscendC::TPipe pipe;

    if constexpr (
        schMode ==
        static_cast<uint32_t>(
            SquareTilingKey::TILING_KEY_SQUARE_MODE_0)) {
        NsSquare::Square<half> op;

        op.Init(
            input_x,
            output,
            &tilingData,
            &pipe);

        op.Process();
    }

    if constexpr (
        schMode ==
        static_cast<uint32_t>(
            SquareTilingKey::TILING_KEY_SQUARE_MODE_1)) {
        NsSquare::Square<float> op;

        op.Init(
            input_x,
            output,
            &tilingData,
            &pipe);

        op.Process();
    }

    (void)workspace;
}