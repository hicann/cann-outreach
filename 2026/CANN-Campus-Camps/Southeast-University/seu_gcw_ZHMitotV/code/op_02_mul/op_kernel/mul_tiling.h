// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;    // 输入张量总元素个数
    uint32_t tileNum;   // 每个Block内部划分的tile数量
};