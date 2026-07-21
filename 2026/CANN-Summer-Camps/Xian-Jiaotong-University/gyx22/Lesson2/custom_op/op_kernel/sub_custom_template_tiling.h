#ifndef SUB_CUSTOM_TEMPLATE_TILING_H
#define SUB_CUSTOM_TEMPLATE_TILING_H

#include <cstdint>

struct SubCustomTemplateTilingData {
    int64_t totalLength;   // 规范要求用 int64_t
    uint32_t tileNum;
    uint32_t dtype;        // 0: float32, 1: float16
};

#endif