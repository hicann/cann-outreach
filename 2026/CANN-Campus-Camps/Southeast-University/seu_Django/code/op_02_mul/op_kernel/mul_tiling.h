#ifndef MUL_TILING_H
#define MUL_TILING_H

#include <cstdint>

// Host/Device 共用，所有长度的单位均为元素。
struct MulTilingData {
    uint32_t totalLength;
    uint32_t tileNum;      // 全局 tile 总数，向上取整；不是固定每核 8 块
    uint32_t tileLength;   // 32 字节对齐的单次处理长度
    uint32_t coreNum;
};

#endif
