#ifndef ADD_CUSTOM_TEMPLATE_TILING_H
#define ADD_CUSTOM_TEMPLATE_TILING_H
#include <cstdint>

struct AddCustomTemplateTilingData {
    uint32_t totalLength;
    uint32_t BlockLength;   // 每核处理量
    uint32_t tileNum;          // 每核 tile 数
    uint32_t tileLength;       // 每个 tile 元素数
    uint32_t blockDim;         // 实际使用的核数
    uint32_t typeSize;         // 数据类型大小
};
#endif 
