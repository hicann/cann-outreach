#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t length;
    uint32_t smallBlockLength;
    uint32_t bigCoreCount;
};
