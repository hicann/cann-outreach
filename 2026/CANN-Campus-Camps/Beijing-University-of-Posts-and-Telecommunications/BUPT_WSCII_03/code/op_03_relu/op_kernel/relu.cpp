#include "relu.h"

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

    NsRelu::Relu<DTYPE_X> op;

    op.Init(
        x,
        y,
        &tilingData);

    op.Process();
}