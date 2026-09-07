// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint64_t totalLength;  // 输入元素总个数
    uint32_t usedCoreNum;  // 实际启动的核数
    uint32_t blockLength;  // 单核处理的元素个数(按32B对齐切分, 最后一个核处理剩余部分)
    uint32_t tileLength;   // UB 单 tile 处理的元素个数(按32B对齐)
};
