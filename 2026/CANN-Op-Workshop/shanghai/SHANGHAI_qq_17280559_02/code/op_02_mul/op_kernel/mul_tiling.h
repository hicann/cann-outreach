// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 输入 x/y 的总元素个数（所有维度累乘）
    uint32_t tileNum;      // 每个核内 blockLength 的分块数
};
