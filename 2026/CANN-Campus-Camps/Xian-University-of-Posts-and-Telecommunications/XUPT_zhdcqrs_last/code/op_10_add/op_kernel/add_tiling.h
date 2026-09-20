// // Tiling结构体定义的头文件
// #pragma once

// #include <cstdint>

// struct AddTilingData {
//     uint32_t length;
// };

#pragma once

#include <cstdint>

struct AddTilingData {
    uint32_t length;       // 输入总元素数
    uint32_t blockFactor;  // 每个核最多处理的元素数
    uint32_t ubFactor;     // 每次搬入 UB 进行计算的元素数
};