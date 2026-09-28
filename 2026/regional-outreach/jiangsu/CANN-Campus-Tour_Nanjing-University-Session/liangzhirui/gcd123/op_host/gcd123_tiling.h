/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * gcd123 算子 Tiling 数据结构定义。
 *
 * 说明：本头文件由算子工程的 host 侧与 kernel 侧共用。
 * Tiling 数据结构全部使用标量字段，避免数组字段在不同 CANN 版本间的兼容性问题。
 */
#ifndef GCD123_TILING_H
#define GCD123_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Gcd123TilingData)
  // 通用切分参数
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);       // 参与计算的核数
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);    // 输出元素总数 = N4*N3*N2*N1
  TILING_DATA_FIELD_DEF(uint32_t, coreLen);        // 每个核处理的元素个数（按真实元素计）
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);       // UB 中单个 tile 的最大元素数（含行尾填充）
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerTile);    // 行对齐模式下每 tile 的行数（N1>=CHUNK 时为 0）
  TILING_DATA_FIELD_DEF(uint32_t, pitch);          // UB 中每行的步长（32B 对齐，= AlignUp(N1,16)）
  TILING_DATA_FIELD_DEF(uint32_t, maxIter);        // GCD 欧几里得迭代最大次数（固定迭代，避免数据相关分支）
  // 输出 shape（4 维，ND 排布，[N4,N3,N2,N1]）
  TILING_DATA_FIELD_DEF(uint32_t, n4);
  TILING_DATA_FIELD_DEF(uint32_t, n3);
  TILING_DATA_FIELD_DEF(uint32_t, n2);
  TILING_DATA_FIELD_DEF(uint32_t, n1);
  // self 的有效 stride（维度广播时置 0；否则为该输入自身连续排布的 stride）
  TILING_DATA_FIELD_DEF(uint32_t, selfS0);
  TILING_DATA_FIELD_DEF(uint32_t, selfS1);
  TILING_DATA_FIELD_DEF(uint32_t, selfS2);
  TILING_DATA_FIELD_DEF(uint32_t, selfS3);
  // other 的有效 stride
  TILING_DATA_FIELD_DEF(uint32_t, otherS0);
  TILING_DATA_FIELD_DEF(uint32_t, otherS1);
  TILING_DATA_FIELD_DEF(uint32_t, otherS2);
  TILING_DATA_FIELD_DEF(uint32_t, otherS3);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Gcd123, Gcd123TilingData)
} // namespace optiling
#endif // GCD123_TILING_H
