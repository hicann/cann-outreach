/**
 * @file atanh_tiling.h
 * @brief Atanh 算子 Tiling 数据结构定义
 *
 * Tiling 数据在 host 侧（tiling 函数）生成、存入 buffer，
 * 通过 GET_TILING_DATA 在 kernel 侧读取，用于指导多核切分。
 */
#ifndef ATANH_TILING_H
#define ATANH_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AtanhTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);  // 输入总元素个数（4 维 shape 展平）
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);     // 实际启动的核数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Atanh, AtanhTilingData);

}  // namespace optiling

#endif  // ATANH_TILING_H
