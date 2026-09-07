// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    int64_t totalNum;        // 总元素数量
    int64_t usedCoreNum;     // 实际使用的核数
    int64_t blockLength;     // 常规核处理的元素数量（32B 对齐）
    int64_t lastBlockLength; // 余数核处理的元素数量（含零头）
    int64_t ubLength;        // 每次 UB 循环处理的元素数量
    int64_t alignNum;        // 32B 对齐所需元素数（fp32=8, fp16=16）
    int64_t extraBlocks;     // 负载均衡：前 extraBlocks 个核各多处理 1 个对齐块
};
