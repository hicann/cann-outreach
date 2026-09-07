// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData 
{
   uint32_t totalLength;  // 输入张量总元素个数
    uint32_t tileLength;   // 每个 tile 处理的元素个数（32 字节对齐）
    uint32_t alignNum;     // 一个 32 字节块包含的元素个数
};