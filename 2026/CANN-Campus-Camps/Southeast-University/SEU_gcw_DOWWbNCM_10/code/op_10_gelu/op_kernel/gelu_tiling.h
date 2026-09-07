// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t totalLength;  // 输入张量元素总个数
    uint32_t blockLength;  // 每个核分摊的元素个数（向上取整，无需 32B 对齐）
    uint32_t tileLength;   // 单核内每个 UB 分块的元素个数
};