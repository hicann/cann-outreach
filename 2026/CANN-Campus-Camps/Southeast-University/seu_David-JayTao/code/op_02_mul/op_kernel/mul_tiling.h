#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 整个 Tensor 元素总数
    uint32_t blockLength;  // 每个 AI Core 处理多少元素
    uint32_t tileNum;      // 每个 Core 的逻辑 tile 数
    uint32_t tileLength;   // 每次真正处理多少元素
};