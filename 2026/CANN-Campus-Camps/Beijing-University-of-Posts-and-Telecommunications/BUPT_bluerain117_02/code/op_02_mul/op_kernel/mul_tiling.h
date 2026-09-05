// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength; // 输入总元素个数
    uint32_t tileNum;     // 每个核内的分块数（每块由BUFFER_NUM个buffer轮转搬运）
};