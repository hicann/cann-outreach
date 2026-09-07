#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t coreNum;
    uint32_t baseBlockCount;
    uint32_t extraBlockCores;
    uint32_t tailElements;
    uint32_t tileLength;
    uint32_t tmpBufferSize;
};