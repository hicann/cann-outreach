/**
 * Copyright (c) 2025. Host-side operator registration for abs_custom.
 *
 * Registers the abs_custom operator with the Ascend Ops framework,
 * defines the tiling strategy, and provides operator attributes.
 */
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include <algorithm>
#include <cmath>

using namespace AscendC;

// ---------------------------------------------------------------------------
// Tiling data (must match the struct in op_kernel/abs_custom.cpp)
// ---------------------------------------------------------------------------
struct AbsCustomTilingData {
    uint32_t totalLength;   // Total number of float16 elements
    uint32_t tileLen;       // Elements per tile
};

// ---------------------------------------------------------------------------
// Operator registration
// ---------------------------------------------------------------------------
IMPLEMT_COMMON(AbsCustom, "abs_custom", INPUTS({ { "x", GE_FLOAT16, TENSOR } }),
               OUTPUTS({ { "y", GE_FLOAT16, TENSOR } }));

/**
 * Tiling function.
 *
 * Given the input shape, determines:
 *   - totalLength: total number of elements
 *   - tileLen:     per-tile element count (fits in UB)
 *
 * For shapes [1,128], [4,2048], [32,4096]:
 *   - [1,128]     -> total=128,   tile=128
 *   - [4,2048]    -> total=8192,  tile=2048 (max for 4KB UB per vector)
 *   - [32,4096]   -> total=131072, tile=2048 (split across tiles)
 */
IMPLEMT_COMMON_TILING(AbsCustom, "abs_custom") {
    auto op = TilingOp("abs_custom", tilingData->blockDim, args->GetInput(0));
    auto &xShape = args->GetInput(0)->GetShape();

    // ---- Calculate total number of elements ----
    uint32_t totalLength = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        totalLength *= xShape.GetDim(i);
    }

    // ---- Determine tile length ----
    // Each tile should fit comfortably in UB (~248KB usable on 910B3).
    // For float16 (2 bytes), 2048 elements = 4KB per vector, very safe.
    // We keep totalLength itself as the maximum tile.
    // Ascend C vector instructions work on 32B-aligned (16 half) chunks.
    uint32_t tileLen = std::min(totalLength, (uint32_t)2048u);
    // Align down to multiple of 16
    tileLen = (tileLen / 16) * 16;
    if (tileLen == 0) tileLen = 16;

    // ---- Serialise tiling data ----
    AbsCustomTilingData tiling = { totalLength, tileLen };
    tilingData->SetRawTiling(&tiling, sizeof(AbsCustomTilingData));

    // ---- Set block dimension ----
    // Use up to 8 blocks (AICores) for efficiency
    uint32_t maxBlocks = 8;
    uint32_t blockDim = std::min(maxBlocks, (totalLength + tileLen - 1) / tileLen);
    if (blockDim == 0) blockDim = 1;
    tilingData->blockDim = blockDim;

    return TilingSuccess;
}

// ---------------------------------------------------------------------------
// Operator implementation registration (links kernel to operator)
// ---------------------------------------------------------------------------
INFER_SHAPE(AbsCustom, "abs_custom") {
    auto &xShape = args->GetInput(0)->GetShape();
    auto *yShape = args->GetOutput(0)->MutShape();
    *yShape = xShape;   // Output shape == input shape
    return GraphSuccess;
}
