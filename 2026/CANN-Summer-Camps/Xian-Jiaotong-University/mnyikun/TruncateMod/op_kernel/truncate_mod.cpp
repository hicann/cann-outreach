/*!
 * \file truncate_mod.cpp
 * \brief TruncateMod 算子 kernel 入口
 */

#include "truncate_mod.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void truncate_mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TruncateModTilingData);
    GET_TILING_DATA_WITH_STRUCT(TruncateModTilingData, tilingData, tiling);
    NsTruncateMod::TruncateMod<D_T_X, BUFFER_MODE> op;
    op.Init(x1, x2, y, &tilingData);
    op.Process();
}
