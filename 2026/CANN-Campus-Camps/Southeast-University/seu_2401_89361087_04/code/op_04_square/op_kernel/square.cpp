/*!
 * \file square.cpp
 * \brief Square kernel entry
 */

#include "square.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"


template <uint32_t schMode>
__global__ __aicore__ void square(
    GM_ADDR inputX,
    GM_ADDR output,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        SquareTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        SquareTilingData,
        tilingData,
        tiling);


    // schMode = 0
    // float16
    if constexpr (schMode == 0) {

        NsSquare::Square<half> op;

        op.Init(
            inputX,
            output,
            &tilingData);

        op.Process();
    }


    // schMode = 1
    // float32
    else if constexpr (schMode == 1) {

        NsSquare::Square<float> op;

        op.Init(
            inputX,
            output,
            &tilingData);

        op.Process();
    }
}
