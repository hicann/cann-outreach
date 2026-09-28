/*
 * Relu 算子 host 侧：tiling 数据结构与生成函数
 * shape 为 4 维 [N4, N3, N2, N1]，ND 格式，内部折算成一维总元素量做分块
 */
#pragma once
#include "kernel_operator.h" // host 侧编译同样使用 AscendC 头

struct ReluTilingData {
    uint32_t totalNum; // N4*N3*N2*N1
    uint32_t tileSize; // 每次搬入 UB 的元素数，与 kernel 的 TILE_SIZE 一致
};

// 由 4 维 shape 计算 tiling
inline void ComputeTiling(const uint32_t shape[4], ReluTilingData &tiling)
{
    tiling.totalNum = shape[0] * shape[1] * shape[2] * shape[3];
    tiling.tileSize = 2048; // 与 kernel/kernel_relu_custom.cpp 中 TILE_SIZE 保持一致
}
