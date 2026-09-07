// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 总元素个数
    uint32_t blockLength;  // 每个核处理的元素个数（保证32B对齐，尾核可能不足）
    uint32_t tileNum;      // 每核内数据的分块个数
    uint32_t tileLength;   // 核内每个分块的元素个数（保证32B对齐，尾块可能不足）
};
