#ifndef RELU_TILING_KEY_H
#define RELU_TILING_KEY_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_KEY_DEF(ReluTilingKey)
    TILING_KEY_FIELD(DTYPE, x, 0)
END_TILING_KEY_DEF;
}
#endif
