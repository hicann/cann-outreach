// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength; // 输入张量总元素个数（x/y/z 一致），本题为 8 * 2048
    uint32_t tileNum;     // 每个核内部分块数（单块方案取 1）
};
