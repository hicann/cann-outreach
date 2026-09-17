/*
 * add_custom 算子 - tiling 数据结构与 host 侧核函数封装
 */
#pragma once
#include "kernel_tiling/kernel_tiling.h"
#include <cstdint>

// 传递给 device 侧的 tiling 参数（与核函数中 GET_TILING_DATA 对应）
struct AddCustomTilingData {
    uint32_t totalLength;  // 张量元素总数 N2 * N1
    uint32_t tileLength;   // 每次搬入/计算的块大小
};

#ifdef __CCE_KT_TEST__
// host 侧编译（CPU 调试）时声明，使核函数可被 host 调用
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR workspace, GM_ADDR tiling);
#endif

namespace AscendC {

// host 侧核函数封装：调用 add_custom
inline void AddCustomDo(const uint32_t blockDim, void *l2ctrl, void *stream,
                        const uint8_t *x, const uint8_t *y, const uint8_t *z,
                        const uint8_t *workspace, const uint8_t *tiling)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}

} // namespace AscendC
