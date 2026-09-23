/*!
 * \file exp.cpp
 * \brief Exp 算子 kernel 入口
 */

#include "exp.h"

template <typename DT_X>
__global__ __aicore__ void exp(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ExpTilingData);
    GET_TILING_DATA_WITH_STRUCT(ExpTilingData, tilingData, tiling);
    // TODO 考生自行补齐
}
