// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// 当前 tiling 方案只需 length（host 侧算核数并 SetBlockDim，kernel 侧 GetBlockNum 取到一致值）
// 若后续 kernel 侧需要更多 tiling 字段，可在此扩展
struct MulTilingData {
    uint32_t length;   // 总元素数（host 侧填入 x 的元素个数）
};
