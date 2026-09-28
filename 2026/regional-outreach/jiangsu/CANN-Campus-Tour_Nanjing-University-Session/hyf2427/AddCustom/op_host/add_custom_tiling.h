/**
 * @file add_custom_tiling.h
 *
 * @brief add_custom 算子 Tiling 参数定义（host 侧）
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength); // 总计算数据量（x 的元素总数 = N2 * N1）
TILING_DATA_FIELD_DEF(uint32_t, tileNum);     // 每个核上总计算数据分块个数
END_TILING_DATA_DEF;

// 注册 tiling 数据到对应的算子
REGISTER_TILING_DATA_CLASS(AddCustom, TilingData)
} // namespace optiling
#endif // ADD_CUSTOM_TILING_H
