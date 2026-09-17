/*
 * @file atanh_custom_tiling.h
 * @brief atanh 算子的 Tiling 数据结构定义（Host 侧与 Kernel 侧共享）
 *
 * 说明：
 *  - atanh 是逐元素一元算子，4 维 ND 张量 [N4, N3, N2, N1] 在内存中连续存放，
 *    因此核函数只需关心"总元素数"，无需关心维度本身。
 *  - 本结构体由 Host 侧 TilingFunc 填充，序列化后通过 GM 传给 Kernel 侧。
 */

#ifndef ATANH_CUSTOM_TILING_H
#define ATANH_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AtanhCustomTilingData)
    // 输入张量总元素数 = N4 * N3 * N2 * N1
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    // 每个核（block）的基准处理长度（向上取整，最后一个核由 Kernel 侧修正）
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
    // 核内单个 tile 的长度（已对齐到 256，保证 DataCopy 32 字节对齐）
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

// 注册 Tiling 数据结构，使框架与 Kernel 侧能够正确反序列化
REGISTER_TILING_DATA_CLASS(AtanhCustom, AtanhCustomTilingData)

}  // namespace optiling

#endif  // ATANH_CUSTOM_TILING_H
