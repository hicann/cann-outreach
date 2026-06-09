#ifndef TANH_CUSTOM_TILING_H
#define TANH_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TanhCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, tailLength);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(TanhCustom, TanhCustomTilingData)
} // namespace optiling

#endif // TANH_CUSTOM_TILING_H
