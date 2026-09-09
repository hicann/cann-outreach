/*!
 * \file tanh.cpp
 * \brief Tanh 算子 kernel 入口
 */

#include "tanh.h"

template <typename DT_X>
__global__ __aicore__ void tanh(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhTilingData);
    GET_TILING_DATA_WITH_STRUCT(TanhTilingData, tilingData, tiling);
    // TODO 考生自行补齐
}
