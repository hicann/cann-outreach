/*!
 * \file square.cpp
 * \brief Square kernel entry
 */

#include "square.h"

enum class SquareTilingKey : uint32_t
{
    TILING_KEY_SQUARE_MODE_0 = 0,
    TILING_KEY_SQUARE_MODE_1 = 1,
};


template <uint32_t schMode>
__global__ __aicore__ void square(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        SquareTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        SquareTilingData,
        tilingData,
        tiling);


    // mode 0 -> float16
    if constexpr (
        schMode ==
        static_cast<uint32_t>(
            SquareTilingKey::
                TILING_KEY_SQUARE_MODE_0)) {

        NsSquare::Square<half> op;

        op.Init(
            x,
            y,
            &tilingData);

        op.Process();
    }


    // mode 1 -> float32
    if constexpr (
        schMode ==
        static_cast<uint32_t>(
            SquareTilingKey::
                TILING_KEY_SQUARE_MODE_1)) {

        NsSquare::Square<float> op;

        op.Init(
            x,
            y,
            &tilingData);

        op.Process();
    }
}