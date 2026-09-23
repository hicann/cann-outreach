#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DivCustomTemplateTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);  // 总元素个数（8*2048=16384）
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);   // 每次循环处理的 tile 长度
  TILING_DATA_FIELD_DEF(uint32_t, dataType);     // 0=float16, 1=float32
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DivCustomTemplate, DivCustomTemplateTilingData)
}