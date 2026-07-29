#ifndef ADD_CUSTOM_TEMPLATE_TILING_H
#define ADD_CUSTOM_TEMPLATE_TILING_H

#include <cstdint>

// Tiling configuration for addition custom operator
struct AddCustomTemplateTilingData {
    // Total number of elements across all dimensions
    uint32_t totalLength;
    // Number of tiles to process per block
    uint32_t tileNum;
};

#endif
