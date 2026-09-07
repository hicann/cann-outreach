// Tiling structure shared by host and kernel.
#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t length;     // total number of elements in the input tensor
    uint32_t blockDim;   // number of AI cores the kernel will be launched with
    uint32_t tileLength; // number of elements processed per loop (aligned to 32B)
};
