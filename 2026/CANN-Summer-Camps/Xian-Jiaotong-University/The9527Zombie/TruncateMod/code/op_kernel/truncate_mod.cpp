/*!
 * \file truncate_mod.cpp
 * \brief TruncateMod kernel entry (Ascend C, A2/A3).
 *
 * The host tiling encodes the compute dtype into a single template argument
 * "schMode" (see truncate_mod_tiling_key.h). The device entry is instantiated
 * per schMode value and dispatches, at compile time, to the matching typed
 * implementation NsTruncateMod::Run<T>.
 */
#include "truncate_mod.h"
#include "truncate_mod_tiling_key.h"

template <uint32_t schMode>
__global__ __aicore__ void truncate_mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TruncateModTilingData);
    GET_TILING_DATA_WITH_STRUCT(TruncateModTilingData, tilingData, tiling);

    if constexpr (schMode == TRUNCATE_MOD_SCH_FP16) {
        NsTruncateMod::Run<half>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATE_MOD_SCH_FP32) {
        NsTruncateMod::Run<float>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATE_MOD_SCH_BF16) {
        NsTruncateMod::Run<bfloat16_t>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATE_MOD_SCH_INT32) {
        NsTruncateMod::Run<int32_t>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATE_MOD_SCH_INT8) {
        NsTruncateMod::Run<int8_t>(x1, x2, y, &tilingData);
    } else if constexpr (schMode == TRUNCATE_MOD_SCH_UINT8) {
        NsTruncateMod::Run<uint8_t>(x1, x2, y, &tilingData);
    }
}
