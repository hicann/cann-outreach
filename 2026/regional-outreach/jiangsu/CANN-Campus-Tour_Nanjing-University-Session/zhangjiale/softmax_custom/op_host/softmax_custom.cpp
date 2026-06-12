/**
 * Copyright (c) 2025. Host-side registration for softmax_custom.
 *
 * Shape: e.g. [128,64,32], dtype: float32, Format: ND, axis=-1.
 *
 * === 验证记录 ===
 * IMPLEMT_COMMON / IMPLEMT_COMMON_TILING / INFER_SHAPE
 * → 对照 abs_custom host 端完全一致的模式 ✓
 */
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include <algorithm>

using namespace AscendC;

// ---------------------------------------------------------------------------
// Tiling data (must match struct in op_kernel/softmax_custom.cpp)
// ---------------------------------------------------------------------------
struct SoftmaxCustomTilingData {
    uint32_t totalRows;
};

// ---------------------------------------------------------------------------
// Operator registration
// ---------------------------------------------------------------------------
IMPLEMT_COMMON(SoftmaxCustom, "softmax_custom",
               INPUTS({ { "x", GE_FLOAT32, TENSOR } }),
               OUTPUTS({ { "y", GE_FLOAT32, TENSOR } }));

/**
 * Tiling: compute totalRows from shape, distribute across AICores.
 *
 * For [128,64,32], axis=-1:
 *   totalRows = 128 × 64 = 8192  (each row = 32 elements)
 *   blockDim  = min(8, 8192) = 8, each core gets ~1024 rows
 */
IMPLEMT_COMMON_TILING(SoftmaxCustom, "softmax_custom") {
    // Ref[abs]: auto op = TilingOp("abs_custom", tilingData->blockDim, ...);
    auto &xShape = args->GetInput(0)->GetShape();
    uint32_t dimNum = xShape.GetDimNum();
    ASSERT(dimNum >= 2 && "softmax needs at least 2D input");

    // Product of all dims except the last = number of rows
    uint32_t totalRows = 1;
    for (size_t i = 0; i < dimNum - 1; i++) {
        totalRows *= xShape.GetDim(i);
    }

    // Serialize tiling data
    SoftmaxCustomTilingData tiling = { totalRows };
    tilingData->SetRawTiling(&tiling, sizeof(SoftmaxCustomTilingData));

    // Block dimension: up to 8 AICores
    uint32_t maxBlocks = 8;
    uint32_t blockDim = std::min(maxBlocks, totalRows);
    if (blockDim == 0) blockDim = 1;
    tilingData->blockDim = blockDim;

    return TilingSuccess;
}

// ---------------------------------------------------------------------------
// Infer shape: output shape == input shape
// ---------------------------------------------------------------------------
INFER_SHAPE(SoftmaxCustom, "softmax_custom") {
    auto &xShape = args->GetInput(0)->GetShape();
    auto *yShape = args->GetOutput(0)->MutShape();
    *yShape = xShape;
    return GraphSuccess;
}
