// Tiling结构体定义的头文件
#pragma once
#include <cstdint>

struct MulTilingData {
    // 完全对齐参考加法算子的传参规范，没有多余字段
    uint32_t totalLength; // 张量总元素个数，比如 8*2048=16384
    uint32_t tileNum;     // 单核内分块的总块数，固定为8
};
