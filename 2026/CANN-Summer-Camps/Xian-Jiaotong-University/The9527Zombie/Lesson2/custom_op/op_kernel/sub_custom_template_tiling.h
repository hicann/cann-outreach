#ifndef SUB_CUSTOM_TEMPLATE_TILING_H
#define SUB_CUSTOM_TEMPLATE_TILING_H
#include <cstdint>

struct SubCustomTemplateTilingData {
    uint64_t totalLength;   // 总元素数 (uint64_t 避免超大 tensor 截断)
    uint32_t blockLength;   // 每个 Block 处理的元素数 = ceil(totalLength / 8)
    uint32_t dataTypeSize;  // 单元素字节数: 2=float16, 4=float32
};

#endif