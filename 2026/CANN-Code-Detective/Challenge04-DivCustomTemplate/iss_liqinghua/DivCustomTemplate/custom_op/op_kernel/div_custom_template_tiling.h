#ifndef DIV_CUSTOM_TEMPLATE_TILING_H
#define DIV_CUSTOM_TEMPLATE_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DivCustomTemplateTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, blockLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
    TILING_DATA_FIELD_DEF(uint32_t, dataType);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DivCustomTemplate, DivCustomTemplateTilingData)

}  // namespace optiling

#endif  // DIV_CUSTOM_TEMPLATE_TILING_H
