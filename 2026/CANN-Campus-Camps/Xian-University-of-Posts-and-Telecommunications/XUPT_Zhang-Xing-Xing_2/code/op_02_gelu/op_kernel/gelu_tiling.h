#pragma once
#include <cstdint>

// Tiling 结构体：v3 起的结构（length / blockDim / tileLength），保持不变
struct GeluTilingData {
    uint32_t length;      // 输入总元素个数
    uint32_t blockDim;    // 启动核数
    uint32_t tileLength;  // 每个 tile 的元素个数
};
