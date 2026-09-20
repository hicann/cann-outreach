// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct AddTilingData {
    uint32_t totalLength;  // 输入张量的元素总个数
    uint32_t tileNum;      // 每个核上的片内分块数
};
