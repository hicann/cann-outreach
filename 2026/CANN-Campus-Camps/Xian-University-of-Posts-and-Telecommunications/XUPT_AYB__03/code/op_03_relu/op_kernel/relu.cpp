#include "kernel_operator.h"
#include "relu.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void relu(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    KernelRelu<DTYPE_X> op;

    op.Init(
        x,
        y,
        tilingData.blockLen,
        tilingData.tileLen,
        tilingData.tileNumPerCore);

    op.Process();
}