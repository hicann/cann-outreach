/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * gcd123 直接调用（kernel_invocation）模式的 Tiling 结构体定义。
 * 供 host 侧 main.cpp 与 gen_data_and_tiling.py 共同遵守。
 *
 * 字段顺序务必与 gen_data_and_tiling.py 中的打包顺序一致！
 */
#ifndef GCD123_TILING_DEF_H
#define GCD123_TILING_DEF_H

#include <cstdint>

struct Gcd123TilingData {
    // 通用切分参数
    uint32_t blockDim;      // 参与计算的核数
    uint32_t totalLength;   // 输出元素总数 = N4*N3*N2*N1
    uint32_t coreLen;       // 每个核处理的元素个数
    uint32_t tileSize;      // UB 中单个 tile 的最大元素数（含行尾填充）
    uint32_t rowsPerTile;   // 行对齐模式下每 tile 的行数（N1>=2048 时为 0）
    uint32_t pitch;         // UB 中每行的步长（32B 对齐，= AlignUp(N1,16)）
    uint32_t maxIter;       // GCD 欧几里得迭代最大次数
    // 输出 shape（4 维，ND 排布，[N4,N3,N2,N1]）
    uint32_t n4;
    uint32_t n3;
    uint32_t n2;
    uint32_t n1;
    // self 的有效 stride（广播维置 0）
    uint32_t selfS0;
    uint32_t selfS1;
    uint32_t selfS2;
    uint32_t selfS3;
    // other 的有效 stride（广播维置 0）
    uint32_t otherS0;
    uint32_t otherS1;
    uint32_t otherS2;
    uint32_t otherS3;
};

#endif // GCD123_TILING_DEF_H
