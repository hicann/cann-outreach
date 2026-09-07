// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t length;    // 数据总长度（元素个数），如 8*2048
    uint32_t tileNum;   // 单核内切分的数据块数（Tiling 粒度）
};