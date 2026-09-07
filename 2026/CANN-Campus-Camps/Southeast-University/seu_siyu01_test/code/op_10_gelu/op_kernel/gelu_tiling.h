#pragma once
#include <cstdint>
struct GeluTilingData {
    uint64_t length;
    uint32_t tile_elements;
    uint32_t block_count;
    uint32_t erf_tmp_bytes;
    uint32_t buffer_count;
    uint64_t core_elements;
    uint32_t extra_core_count;
};
