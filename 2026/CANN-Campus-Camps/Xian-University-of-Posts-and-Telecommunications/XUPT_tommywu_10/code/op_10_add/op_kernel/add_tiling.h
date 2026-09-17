#ifndef ADD_TILING_H
#define ADD_TILING_H

#include <cstdint>

struct AddTilingData {
    uint32_t totalLength;  // 输入/输出总元素个数
    uint32_t tileNum;      // 每个核上需要循环处理的块数
};

#endif  // ADD_TILING_H