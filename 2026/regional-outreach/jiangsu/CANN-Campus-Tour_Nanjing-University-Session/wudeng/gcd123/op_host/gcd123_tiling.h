#ifndef GCD123_TILING_H
#define GCD123_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t GCD123_NDIM = 4;

BEGIN_TILING_DATA_DEF(Gcd123TilingData)
    TILING_DATA_FIELD_DEF(int64_t, selfLength);
    TILING_DATA_FIELD_DEF(int64_t, otherLength);
    TILING_DATA_FIELD_DEF(int64_t, totalLength);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 4, selfShape);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 4, otherShape);
    TILING_DATA_FIELD_DEF_ARR(int64_t, 4, outShape);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(gcd123, Gcd123TilingData)

}  // namespace optiling

#endif  // GCD123_TILING_H
