/**
 * Copyright (c) 2025. Host-side operator registration for add_custom.
 *
 * Registers the add_custom operator with the Ascend Ops framework,
 * defines the tiling strategy, and provides operator attributes.
 *
 * Inputs:  x (float16, ND), y (float16, ND)
 * Outputs: z (float16, ND)
 * Math:    z = x + y  (element-wise)
 */
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include <algorithm>

using namespace AscendC;

// ---------------------------------------------------------------------------
// Tiling data (must match the struct in op_kernel/add_custom.cpp)
// ---------------------------------------------------------------------------
struct AddCustomTilingData {
    uint32_t totalLength;   // Total number of float16 elements
    uint32_t tileLen;       // Elements per tile
};

// ---------------------------------------------------------------------------
// Operator registration
//   Name:       add_custom
//   Inputs:     x (float16, tensor), y (float16, tensor)
//   Outputs:    z (float16, tensor)
// ---------------------------------------------------------------------------
IMPLEMT_COMMON(AddCustom, "add_custom",
    INPUTS({ { "x", GE_FLOAT16, TENSOR },
             { "y", GE_FLOAT16, TENSOR } }),
    OUTPUTS({ { "z", GE_FLOAT16, TENSOR } }));

/**
 * Tiling function.
 *
 * Given two input tensors with identical shape [N2, N1], determines:
 *   - totalLength: N2 × N1 (total number of elements)
 *   - tileLen:     per-tile element count (fits in UB)
 *
 * For example shapes:
 *   [1, 128]     -> total=128,   tile=128   (1 tile)
 *   [4, 2048]    -> total=8192,  tile=2048  (4 tiles per block)
 *   [32, 4096]   -> total=131072, tile=2048 (64 tiles per block)
 *
 * The tile length is capped at 2048 elements (4 KB for float16) so that
 * double-buffered 2-input + 1-output fit comfortably in UB (~248 KB).
 */
IMPLEMT_COMMON_TILING(AddCustom, "add_custom") {
    auto op = TilingOp("add_custom", tilingData->blockDim, args->GetInput(0));
    auto &xShape = args->GetInput(0)->GetShape();
    auto &yShape = args->GetInput(1)->GetShape();

    // ---- Validate shapes ----
    ASSERT(xShape.GetDimNum() == yShape.GetDimNum() &&
           "x and y must have the same number of dimensions");

    // ---- Calculate total number of elements ----
    uint32_t totalLength = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        ASSERT(xShape.GetDim(i) == yShape.GetDim(i) &&
               "x and y must have the same shape");
        totalLength *= xShape.GetDim(i);
    }

    // ---- Determine tile length ----
    // Cap at 2048 elements (4 KB for half) so UB buffers fit safely.
    uint32_t tileLen = std::min(totalLength, (uint32_t)2048u);
    // Align down to multiple of 16 (half-floats) = 32B vector boundary
    tileLen = (tileLen / 16) * 16;
    if (tileLen == 0) tileLen = 16;

    // ---- Serialise tiling data ----
    AddCustomTilingData tiling = { totalLength, tileLen };
    tilingData->SetRawTiling(&tiling, sizeof(AddCustomTilingData));

    // ---- Set block dimension ----
    // Use up to 8 AICores; at least one tile per block
    uint32_t maxBlocks = 8;
    uint32_t minTilesPerBlock = 1;
    uint32_t numTiles = (totalLength + tileLen - 1) / tileLen;
    uint32_t blockDim = std::min(maxBlocks, numTiles / minTilesPerBlock);
    if (blockDim == 0) blockDim = 1;
    tilingData->blockDim = blockDim;

    return TilingSuccess;
}

// ---------------------------------------------------------------------------
// Shape inference: output shape == input shape
// ---------------------------------------------------------------------------
INFER_SHAPE(AddCustom, "add_custom") {
    auto &xShape = args->GetInput(0)->GetShape();
    auto *zShape = args->GetOutput(0)->MutShape();
    *zShape = xShape;
    return GraphSuccess;
}
