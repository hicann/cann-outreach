/*!
 * \file truncate_mod.cpp
 * \brief TruncateMod 算子 kernel 入口
 *
 * 三路 if constexpr 分派（DESIGN.md §3.5）：
 *  - MODE_0：FP16（T=half，升 FP32 计算）
 *  - MODE_1：BF16（T=bfloat16_t，升 FP32 计算）
 *  - MODE_2：FP32（T=float，直算）
 */

#include "truncate_mod.h"

enum class TruncateModTilingKey : uint32_t
{
    TILING_KEY_TRUNCATEMOD_MODE_0 = 0,
    TILING_KEY_TRUNCATEMOD_MODE_1 = 1,
    TILING_KEY_TRUNCATEMOD_MODE_2 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void truncate_mod(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TruncateModTilingData);
    GET_TILING_DATA_WITH_STRUCT(TruncateModTilingData, tilingData, tiling);
    if constexpr (schMode == static_cast<uint32_t>(TruncateModTilingKey::TILING_KEY_TRUNCATEMOD_MODE_0)) {
        NsTruncateMod::TruncateMod<half> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(TruncateModTilingKey::TILING_KEY_TRUNCATEMOD_MODE_1)) {
        NsTruncateMod::TruncateMod<bfloat16_t> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(TruncateModTilingKey::TILING_KEY_TRUNCATEMOD_MODE_2)) {
        NsTruncateMod::TruncateMod<float> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    }
}
