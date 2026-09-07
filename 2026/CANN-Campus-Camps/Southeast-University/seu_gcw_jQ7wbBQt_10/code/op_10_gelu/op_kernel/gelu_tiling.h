// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t totalLength;    // 输入张量元素总个数
    uint32_t ALIGN_NUM;      // 32字节对齐对应的元素个数（fp16=16, fp32=8）
    uint32_t blockSize;      // 单个tile处理的元素个数
    uint32_t coreSize;       // 每个核基础处理的元素个数（按ALIGN_NUM对齐）
    uint32_t extraCores;     // 多分ALIGN_NUM个元素的前几个核的个数
    uint32_t lastRemainder;  // 最后一个核额外处理的不足ALIGN_NUM的余数
};
