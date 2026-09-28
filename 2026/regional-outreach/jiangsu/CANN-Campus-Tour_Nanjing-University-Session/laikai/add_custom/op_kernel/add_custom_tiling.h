#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H

#include <cstdint>

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AddCustomTilingData)
TILING_DATA_FIELD_DEF(uint32_t, totalLength);
TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, AddCustomTilingData)

}  // namespace optiling

#endif  // ADD_CUSTOM_TILING_H
