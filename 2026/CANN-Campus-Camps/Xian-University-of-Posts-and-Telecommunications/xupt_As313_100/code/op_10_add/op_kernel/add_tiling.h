// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct AddTilingData {
    uint32_t totalLength;   // 输入张量的总元素个数
    uint32_t formerNum;     // 处理"大块"的核数(排在前面), 其余核处理"小块"
    uint32_t formerLength;  // 大块长度(元素个数, 512B对齐的整数倍)
    uint32_t tailLength;    // 小块长度(元素个数, 512B对齐的整数倍)
    uint32_t tileLength;    // 核内单次搬运/计算的元素个数, 由UB容量推导
    uint32_t blockDim;      // 实际启动的核数
};
