#ifndef SUB_CUSTOM_TEMPLATE_TILING_H
#define SUB_CUSTOM_TEMPLATE_TILING_H
#include <cstdint>

struct SubCustomTemplateTilingData {
    uint32_t size;         // 总数据量
    uint32_t tileNum;      // Tile数量
    uint32_t tileLength;   // 每个Tile处理的数据量
};

#endif
