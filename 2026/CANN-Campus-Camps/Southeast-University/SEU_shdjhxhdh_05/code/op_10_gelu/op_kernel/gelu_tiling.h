// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    // 输入张量总元素数量
    uint32_t totalLength;

    // 每个普通 Core 处理的元素数量
    // 按 32 Byte 对齐
    uint32_t blockLength;

    // 最后一个 Core 实际处理的元素数量
    uint32_t lastBlockLength;

    // 每次从 GM 搬入 UB 的元素数量
    uint32_t tileLength;
};