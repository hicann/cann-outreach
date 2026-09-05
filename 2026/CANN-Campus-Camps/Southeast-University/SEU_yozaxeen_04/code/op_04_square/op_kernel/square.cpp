/*!
 * \file square.cpp
 * \brief Square optimized kernel entry.
 */

#include "square.h"

enum class SquareTilingKey : uint32_t
{
    TILING_KEY_SQUARE_MODE_0 = 0,
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
    GET_TILING_DATA_WITH_STRUCT(SquareTilingData, tilingData, tiling);
    (void)workspace;

    if constexpr (schMode == static_cast<uint32_t>(SquareTilingKey::TILING_KEY_SQUARE_MODE_0)) {
        NsSquare::SquareExtremeImpl<half>(input_x, output, &tilingData);
    }
    if constexpr (schMode == static_cast<uint32_t>(SquareTilingKey::TILING_KEY_SQUARE_MODE_1)) {
        NsSquare::SquareExtremeImpl<float>(input_x, output, &tilingData);
    }
}
