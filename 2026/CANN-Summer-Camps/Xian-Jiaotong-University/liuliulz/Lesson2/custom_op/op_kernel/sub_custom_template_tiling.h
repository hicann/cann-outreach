#ifndef SUB_CUSTOM_TEMPLATE_TILING_H
#define SUB_CUSTOM_TEMPLATE_TILING_H
#include <cstdint>

struct SubCustomTemplateTilingData {
    uint32_t size;       // 总元素个数  = 8 * 2048 = 16384
    uint32_t dtypeFlag;  // 0 = float16 (half), 1 = float
};

#endif