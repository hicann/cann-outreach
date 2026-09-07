// Tiling结构体定义的头文件5
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint64_t length;       // 所有维度的元素总数，不仅是最后一维 N
    uint64_t blockFactor;  // 每核元素数，起点按 32B 对齐
    uint32_t ubFactor;     // 每次 UB 循环处理的元素个数（块大小）
    uint32_t erfcTmpBytes;  // Erfc<float> 专用临时空间，单位 Byte
};

// 纯整数分块逻辑，Host 使用；也可在无 CANN 的 CPU 上验证边界。
struct GeluBlockPlan {
    uint64_t blockFactor;
    uint32_t blockDim;
};

// 性能调参上限；实际 tile 仍由设备 UB 和 Erfc scratch 预算决定。
constexpr uint32_t GELU_MAX_TILE_ELEMENTS = 4096;

inline GeluBlockPlan PlanGeluBlocks(uint64_t length, uint32_t elementBytes, uint32_t cores)
{
    // 所有数据均以 float 计算，CompareScalar 的最小完整向量是 64 元素。
    // 按此粒度分核避免每核只分到 8/16 个元素却执行 64 个元素的计算。
    (void)elementBytes;
    const uint64_t align = 64;
    const uint64_t blocks = (length + align - 1) / align;
    const uint64_t available = cores == 0 ? 1 : cores;
    const uint64_t wanted = blocks < available ? blocks : available;
    if (wanted == 0) {
        return {align, 1};
    }
    const uint64_t blockFactor = ((blocks + wanted - 1) / wanted) * align;
    return {blockFactor, static_cast<uint32_t>((length + blockFactor - 1) / blockFactor)};
}


