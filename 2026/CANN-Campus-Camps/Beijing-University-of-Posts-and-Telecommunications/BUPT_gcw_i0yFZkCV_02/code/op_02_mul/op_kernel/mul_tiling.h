// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MulTilingData {
    uint32_t totalLength;  // 数据总长度（元素个数），本题 8 * 2048 = 16384
    uint32_t blockLength;  // 每核处理的元素数（核间切分结果）
    uint32_t tileNum;      // 单核内记账组数，物理循环次数 = tileNum * BUFFER_NUM
    uint32_t tileLength;   // 每轮进 UB 的元素数（核内切分结果）
};
