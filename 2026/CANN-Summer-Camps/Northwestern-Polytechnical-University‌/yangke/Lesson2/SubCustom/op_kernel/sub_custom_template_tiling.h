#ifndef SUB_CUSTOM_TEMPLATE_TILING_DATA_H__
#define SUB_CUSTOM_TEMPLATE_TILING_DATA_H__

#include "cstdint"

struct SubCustomTemplateTilingData {
    uint64_t totalLength;
    uint64_t formerLength;
    uint64_t tailLength;
    uint32_t tileLength;
    uint32_t formerTileNum;
    uint32_t tailTileNum;
    uint32_t formerLastTileLength;
    uint32_t tailLastTileLength;
};

#endif // SUB_CUSTOM_TEMPLATE_TILING_DATA_H__
