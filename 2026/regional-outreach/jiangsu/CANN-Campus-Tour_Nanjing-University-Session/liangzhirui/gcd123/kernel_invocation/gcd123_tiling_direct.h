/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * gcd123 直接调用（kernel_invocation）模式的 device 侧 Tiling 读取。
 * 参考官方 Add_tile 样例：直接调用模式下不依赖算子工程的 tiling 注册机制，
 * 用 POD 结构体 + GM->UB 拷贝自行解析 tiling 数据。
 */
#ifndef GCD123_TILING_DIRECT_H
#define GCD123_TILING_DIRECT_H

#include "kernel_invocation/gcd123_tiling_def.h"

#if defined(__CCE_KT_TEST__) || defined(ASCENDC_CPU_DEBUG)
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

inline __aicore__ int32_t AlignDiv32(int32_t n)
{
    return ((n + 31) & ~31) / 32;
}

#define CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    __ubuf__ tilingStruct *tilingDataPointer =                              \
        reinterpret_cast<__ubuf__ tilingStruct *>((__ubuf__ uint8_t *)(tilingPointer));

#if defined(__CCE_KT_TEST__) || defined(ASCENDC_CPU_DEBUG)
#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer) \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer);
#else
#define INIT_TILING_DATA(tilingStruct, tilingDataPointer, tilingPointer)                        \
    __ubuf__ uint8_t* tilingUbPointer = (__ubuf__ uint8_t*)get_imm(0);                          \
    copy_gm_to_ubuf(((__ubuf__ uint8_t*)(tilingUbPointer)), ((__gm__ uint8_t*)(tilingPointer)), \
        0, 1, AlignDiv32(sizeof(tilingStruct)), 0, 0);                                          \
    CONVERT_TILING_DATA(tilingStruct, tilingDataPointer, tilingUbPointer);                      \
    pipe_barrier(PIPE_ALL);
#endif

#define GET_TILING_DATA(tilingData, tilingPointer)                              \
    Gcd123TilingData tilingData;                                                \
    INIT_TILING_DATA(Gcd123TilingData, tilingDataPointer, tilingPointer);       \
    (tilingData).blockDim = tilingDataPointer->blockDim;                        \
    (tilingData).totalLength = tilingDataPointer->totalLength;                  \
    (tilingData).coreLen = tilingDataPointer->coreLen;                          \
    (tilingData).tileSize = tilingDataPointer->tileSize;                        \
    (tilingData).rowsPerTile = tilingDataPointer->rowsPerTile;                  \
    (tilingData).pitch = tilingDataPointer->pitch;                              \
    (tilingData).maxIter = tilingDataPointer->maxIter;                          \
    (tilingData).n4 = tilingDataPointer->n4;                                    \
    (tilingData).n3 = tilingDataPointer->n3;                                    \
    (tilingData).n2 = tilingDataPointer->n2;                                    \
    (tilingData).n1 = tilingDataPointer->n1;                                    \
    (tilingData).selfS0 = tilingDataPointer->selfS0;                            \
    (tilingData).selfS1 = tilingDataPointer->selfS1;                            \
    (tilingData).selfS2 = tilingDataPointer->selfS2;                            \
    (tilingData).selfS3 = tilingDataPointer->selfS3;                            \
    (tilingData).otherS0 = tilingDataPointer->otherS0;                          \
    (tilingData).otherS1 = tilingDataPointer->otherS1;                          \
    (tilingData).otherS2 = tilingDataPointer->otherS2;                          \
    (tilingData).otherS3 = tilingDataPointer->otherS3;

#endif // GCD123_TILING_DIRECT_H
