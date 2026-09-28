#ifndef ATANH_CUSTOM_TILING_H
#define ATANH_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(AtanhCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockNum);
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, numPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, tmpBufferSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AtanhCustom, AtanhCustomTilingData)

}

#endif
