/*!
 * \file sigmoid.cpp
 * \brief Sigmoid 算子 kernel 入口
 */

#include "sigmoid.h"

template <typename DT_X>
__global__ __aicore__ void sigmoid(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SigmoidTilingData);
    GET_TILING_DATA_WITH_STRUCT(SigmoidTilingData, tilingData, tiling);
    // TODO 考生自行补齐
}
