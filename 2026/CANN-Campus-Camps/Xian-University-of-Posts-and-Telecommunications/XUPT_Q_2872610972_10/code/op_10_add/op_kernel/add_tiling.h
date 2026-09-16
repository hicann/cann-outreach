// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct AddTilingData {
    uint32_t totalLength;   // 输入张量总元素个数
    uint32_t tileLength;    // 单次搬运/计算的tile元素个数
    uint32_t coreNum;       // 启动的AI核数
};