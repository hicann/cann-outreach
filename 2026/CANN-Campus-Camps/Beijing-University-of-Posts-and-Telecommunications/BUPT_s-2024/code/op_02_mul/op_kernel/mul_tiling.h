// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// TQue 队列深度（double buffer，流水并行优化）
constexpr int32_t BUFFER_NUM = 2;
// 启动核数：输入shape(8,2048)共16384个元素，8核均分，每核2048个
constexpr uint32_t BLOCK_DIM = 8;
// 每个核上的数据分块个数
constexpr uint32_t TILE_NUM = 8;

struct MulTilingData {
    uint32_t length;    // 总元素个数（8*2048=16384）
    uint32_t tileNum;   // 每个核上总计算数据分块个数
};