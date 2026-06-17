#ifndef TANH_CUSTOM_TILING_H
#define TANH_CUSTOM_TILING_H
#include <cstdint>

struct TanhCustomTilingData {
    uint32_t totalLength;   // 每个核处理的元素个数（总元素数 / BLOCK_DIM）
    uint32_t tileNum;       // 每个核的tile数量，用于双缓冲分片计算
};
#endif
