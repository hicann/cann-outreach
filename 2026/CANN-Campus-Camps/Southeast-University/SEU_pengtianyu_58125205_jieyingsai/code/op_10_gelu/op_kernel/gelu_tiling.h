#pragma once

#include <cstdint>

struct GeluTilingData {
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileLength;

    // Erf使用的独立临时空间大小，单位：字节
    uint32_t erfTmpBytes;
};