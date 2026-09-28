/**
 * @file add_custom_tiling.h
 * @brief add_custom 算子的 TilingData 结构定义。
 *
 * 该头文件同时被 op_host（计算 tiling）和 op_kernel（读取 tiling）引用，
 * 两侧必须使用完全一致的字段定义与顺序，否则会出现静默的数据错位。
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

/**
 * Tiling 字段说明（全部使用 uint32_t，总大小 24 字节，远小于 tiling 空间上限）：
 *
 *   rows         : 输入张量第 0 维长度 N2
 *   cols         : 输入张量第 1 维长度 N1
 *   totalLength  : 元素总数 = rows * cols
 *   tileNum      : 单个 UB tile 的元素个数，恒为 16（32B）的整数倍
 *   tileCount    : tile 总数 = ceil(totalLength / tileNum)
 *   perCoreTile  : 每个 AI Core 分到的 tile 个数（向上取整，尾部核可能不满）
 *
 * 之所以按 "tile" 而不是按 "元素" 给各核分派任务，是因为 tile 粒度天然满足
 * 32B 对齐：每个 tile 的起始地址都落在 32B 边界上，只有最后一个 tile 的长度
 * 可能不足 32B，交给 DataCopyPad 处理即可，无需再做非对齐的头部/尾部特判。
 */
BEGIN_TILING_DATA_DEF(AddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rows);
  TILING_DATA_FIELD_DEF(uint32_t, cols);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileCount);
  TILING_DATA_FIELD_DEF(uint32_t, perCoreTile);
END_TILING_DATA_DEF;

// 将 TilingData 结构注册到算子类型 AddCustom 上。
// 第一个参数必须与 OpDef 的算子类型名一致，否则 kernel 侧 GET_TILING_DATA 取不到结构体。
REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)

}  // namespace optiling

#endif  // ADD_CUSTOM_TILING_H
