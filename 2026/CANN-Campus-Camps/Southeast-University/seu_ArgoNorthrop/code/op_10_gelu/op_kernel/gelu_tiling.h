// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t length;    // 总元素数（host 侧填入 input_x 的元素个数）
    uint32_t perCore;   // 每核处理元素数（host 算好并已 32B 对齐，kernel 免除法）
};