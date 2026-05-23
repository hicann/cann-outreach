#pragma once

#include <cstdint>

struct TrilTilingData {
    uint32_t totalLength;
    uint32_t M;
    uint32_t N;
    uint32_t batchSize;
    int32_t diagonal;
    uint32_t totalRows;
    uint32_t rowsPerCore;
    uint32_t tailRows;
};
