// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct GeluTilingData {
    // Total number of elements in the flattened ND tensor.
    uint64_t totalLength;
    // Number of elements assigned to a normal AIV core.  The last core may
    // process fewer elements.  This value is always 32-byte aligned.
    uint64_t blockLength;
    // Number of elements transferred to UB per iteration.
    uint32_t tileLength;
    // Explicit padding keeps the host/device tiling layout deterministic.
    uint32_t reserved;
};