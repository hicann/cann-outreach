/*!
 * \file add.cpp
 * \brief Add 算子 kernel 入口
 */

#include "add.h"

template <typename DT_X>
__global__ __aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddTilingData, tilingData, tiling);
    NsAdd::Add<DT_X> op;
    op.Init(x, y, z, &tilingData);
    op.Process();
}
