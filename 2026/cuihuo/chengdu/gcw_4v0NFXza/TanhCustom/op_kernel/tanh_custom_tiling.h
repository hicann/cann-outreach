/**
 * @file tanh_custom_tiling.h
 *
 * TanhCustom tiling struct, using CANN standard tiling macros
 * aligned with AddCustom sample pattern (class name = TilingData)
 */
#ifndef TANH_CUSTOM_TILING_H
#define TANH_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TanhCustom, TilingData)
} // namespace optiling
#endif // TANH_CUSTOM_TILING_H
