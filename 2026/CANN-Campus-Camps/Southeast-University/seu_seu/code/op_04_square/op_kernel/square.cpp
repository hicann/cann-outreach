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
    REGISTER_TILING_DEFAULT(
        SquareTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        SquareTilingData,
        tilingData,
        tiling);

    KERNEL_TASK_TYPE_DEFAULT(
        KERNEL_TYPE_AIV_ONLY);

    AscendC::InitSocState();

    (void)workspace;

    if constexpr (
        schMode ==
        static_cast<uint32_t>(
            SquareTilingKey::
                TILING_KEY_SQUARE_MODE_0)) {

        NsSquare::Square<half> op;

        op.Init(
            input_x,
            output,
            &tilingData);

        op.Process();
    }

    if constexpr (
        schMode ==
        static_cast<uint32_t>(
            SquareTilingKey::
                TILING_KEY_SQUARE_MODE_1)) {

        NsSquare::Square<float> op;

        op.Init(
            input_x,
            output,
            &tilingData);

        op.Process();
    }
}
