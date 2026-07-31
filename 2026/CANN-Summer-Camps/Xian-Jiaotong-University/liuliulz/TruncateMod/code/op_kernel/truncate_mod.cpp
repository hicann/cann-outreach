/*!
 * \file truncate_mod.cpp
 * \brief TruncateMod kernel entry — fp16 / fp32 / bf16
 */
#include "truncate_mod.h"

template <uint32_t schMode>
__global__ __aicore__ void truncate_mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y,
                                        GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TruncateModTilingData);
    GET_TILING_DATA_WITH_STRUCT(TruncateModTilingData, tilingData, tiling);

    if constexpr (schMode == TRUNCATEMOD_TPL_SCH_MODE_FP16) {
        NsTruncateMod::TruncateMod<half> op;
        op.Init(x1, x2, y, &tilingData); op.Process();
    }
    if constexpr (schMode == TRUNCATEMOD_TPL_SCH_MODE_FP32) {
        NsTruncateMod::TruncateMod<float> op;
        op.Init(x1, x2, y, &tilingData); op.Process();
    }
    if constexpr (schMode == TRUNCATEMOD_TPL_SCH_MODE_BF16) {
        NsTruncateMod::TruncateMod<bfloat16_t> op;
        op.Init(x1, x2, y, &tilingData); op.Process();
    }
}
