// Tiling结构体定义的头文件
#pragma once
#include <cstdint>

struct AddTilingData {
    uint32_t length;   // 张量总元素个数
    uint32_t perCore;  // 每个核需要处理的元素个数
};
