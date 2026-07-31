/*!
 * \file truncate_div.cpp
 * \brief TruncateDiv 算子 kernel 入口
 *
 * 通过模板参数 schMode 区分不同数据类型：
 *   - Mode 0 (TILING_KEY_TRUNCATEDIV_MODE_0): fp16 (half 类型)
 *   - Mode 1 (TILING_KEY_TRUNCATEDIV_MODE_1): fp32 (float 类型)
 *   - Mode 2 (TILING_KEY_TRUNCATEDIV_MODE_2): bf16 (bfloat16_t 类型,
 *     内部 Cast 到 fp32 计算，因 Div API 不支持 bfloat16_t)
 */

#include "truncate_div.h"

enum class TruncateDivTilingKey : uint32_t
{
    TILING_KEY_TRUNCATEDIV_MODE_0 = 0,
    TILING_KEY_TRUNCATEDIV_MODE_1 = 1,
    TILING_KEY_TRUNCATEDIV_MODE_2 = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void truncate_div(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    // 注册 tiling 数据结构并获取 tiling 数据
    REGISTER_TILING_DEFAULT(TruncateDivTilingData);
    GET_TILING_DATA_WITH_STRUCT(TruncateDivTilingData, tilingData, tiling);

    // 根据 schMode 选择不同的数据类型实例化算子
    if constexpr (schMode == static_cast<uint32_t>(TruncateDivTilingKey::TILING_KEY_TRUNCATEDIV_MODE_0)) {
        // fp16 路径
        NsTruncateDiv::TruncateDiv<half> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    }

    if constexpr (schMode == static_cast<uint32_t>(TruncateDivTilingKey::TILING_KEY_TRUNCATEDIV_MODE_1)) {
        // fp32 路径
        NsTruncateDiv::TruncateDiv<float> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    }

    if constexpr (schMode == static_cast<uint32_t>(TruncateDivTilingKey::TILING_KEY_TRUNCATEDIV_MODE_2)) {
        // bf16 路径：GM 数据为 bfloat16_t，内部 Cast 到 fp32 计算后 Cast 回 bf16
        NsTruncateDiv::TruncateDiv<bfloat16_t> op;
        op.Init(x1, x2, y, &tilingData);
        op.Process();
    }
}
