#ifndef DIV_CUSTOM_TEMPLATE_TILING_H
#define DIV_CUSTOM_TEMPLATE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// Tiling 数据结构：kernel 侧与 host 侧通过该结构传递切分参数
BEGIN_TILING_DATA_DEF(TilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);  // 输入总长度（本题为 8*2048 = 16384）
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);   // 每次循环处理的 tile 长度
    TILING_DATA_FIELD_DEF(uint32_t, dataType);     // 数据类型标记：0 = float16，1 = float32
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DivCustomTemplate, TilingData)

}  // namespace optiling

#endif  // DIV_CUSTOM_TEMPLATE_TILING_H
