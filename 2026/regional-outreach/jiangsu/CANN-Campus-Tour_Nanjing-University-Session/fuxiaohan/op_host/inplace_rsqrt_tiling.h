/**
 * @file inplace_rsqrt_tiling.h
 * @brief InplaceRsqrt 的 TilingData 定义（host 与 kernel 共用）
 */
#ifndef INPLACE_RSQRT_TILING_H
#define INPLACE_RSQRT_TILING_H

#include <stdint.h>
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InplaceRsqrtTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);  // 元素总个数（4 维 shape 展平）
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);   // 单次搬运的元素个数（32B 对齐，如 8192）
END_TILING_DATA_DEF

// 注册：算子类型 InplaceRsqrt 对应 InplaceRsqrtTilingData
REGISTER_TILING_DATA_CLASS(InplaceRsqrt, InplaceRsqrtTilingData)
}  // namespace optiling

#endif  // INPLACE_RSQRT_TILING_H