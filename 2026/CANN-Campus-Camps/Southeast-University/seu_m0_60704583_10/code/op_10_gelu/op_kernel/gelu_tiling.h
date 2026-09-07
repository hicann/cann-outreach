// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t length;    // 输入元素总数
    uint32_t core_num;  // 实际启动的 AIV 核数
};