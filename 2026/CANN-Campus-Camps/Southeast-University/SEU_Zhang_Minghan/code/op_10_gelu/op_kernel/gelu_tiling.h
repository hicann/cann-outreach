// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// GELU 算子 tiling 参数。
// 数据按 32B(1 个数据块)粒度切分后均分到各 AI Core:
//   前 tailBlockNum 个核(大核)多承担 1 个块, 其余核(小核)承担相同的块数,
// 从而保证每个核的起始偏移和处理长度均为 32B 对齐, 兼容元素个数非 32B 对齐的场景。
struct GeluTilingData {
    uint32_t length;            // 输入总元素个数
    uint32_t smallCoreDataNum;  // 小核承担的元素个数
    uint32_t bigCoreDataNum;    // 大核承担的元素个数
    uint32_t finalSmallTileNum; // 小核的分块循环次数
    uint32_t finalBigTileNum;   // 大核的分块循环次数
    uint32_t tileDataNum;       // 单个 tile 的元素个数
    uint32_t smallTailDataNum;  // 小核最后一个 tile 的元素个数
    uint32_t bigTailDataNum;    // 大核最后一个 tile 的元素个数
    uint32_t tailBlockNum;      // 大核个数
};