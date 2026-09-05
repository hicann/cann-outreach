#ifndef RELU_TILING_DATA_H
#define RELU_TILING_DATA_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReluTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalElem);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, blockLen);
    TILING_DATA_FIELD_DEF(uint32_t, tileNumPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tileLen);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(Relu, ReluTilingData)
}
#endif
