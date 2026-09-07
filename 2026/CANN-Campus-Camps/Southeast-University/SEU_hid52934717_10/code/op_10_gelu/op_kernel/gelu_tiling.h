// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t length;
};

// Shared by host tiling and kernel processing so block partitioning cannot
// diverge from the actual UB tile size.
// 8192 elements balance UB setup with AIV parallelism: the largest valid
// tensor (N=10240) still exposes five independent tiles instead of collapsing
// the whole workload onto one core.
static constexpr uint32_t GELU_TILE_LENGTH = 8192;