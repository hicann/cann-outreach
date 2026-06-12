/**
 * Copyright (c) 2025. Host-side operator registration for gcd_custom.
 *
 * Registers the gcd_custom operator with the Ascend Ops framework,
 * defines the tiling strategy, and computes broadcast mapping.
 *
 * Operator: out = gcd(self, other)  (float16, 4D ND, with broadcast)
 *
 * Broadcast semantics for 4D shapes [N4,N3,N2,N1]:
 *   - Each dimension of self and other must be equal or one of them must be 1.
 *   - Output shape takes the max of each dimension.
 *   - A dimension with size 1 is broadcast (stride=0 in kernel mapping).
 */
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/tiling_api.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

using namespace AscendC;

// ---------------------------------------------------------------------------
// Tiling data (must match the struct in op_kernel/gcd_custom.cpp)
// ---------------------------------------------------------------------------
struct GcdCustomTilingData {
    uint32_t totalLength;       // Total output elements (after broadcast)
    uint32_t tileLen;           // Elements per tile
    int32_t selfStrides[4];     // Self strides (0 for broadcast dims)
    int32_t otherStrides[4];    // Other strides (0 for broadcast dims)
    int32_t outStrides[4];      // Output strides for 4D->1D mapping
};

// ---------------------------------------------------------------------------
// Helper: compute broadcast of two 4D shapes
// ---------------------------------------------------------------------------
static void BroadcastShape4D(const std::vector<int64_t> &s1,
                              const std::vector<int64_t> &s2,
                              int64_t outShape[4]) {
    // Assume both shapes are 4D (padded to 4D if needed by caller)
    for (int32_t d = 0; d < 4; d++) {
        outShape[d] = std::max(s1[d], s2[d]);
    }
}

// ---------------------------------------------------------------------------
// Helper: compute strides with broadcast support
//   If shape[d] == 1, stride[d] = 0 (broadcast — always access element 0)
//   Otherwise, stride[d] = product of later dims in shape
// ---------------------------------------------------------------------------
static void ComputeBroadcastStrides(const int64_t shape[4],
                                     int32_t strides[4]) {
    int32_t stride = 1;
    // Compute strides from innermost (N1) to outermost (N4)
    for (int32_t d = 3; d >= 0; d--) {
        strides[d] = (shape[d] == 1) ? 0 : stride;
        if (shape[d] >= 1) {
            stride *= (int32_t)shape[d];
        }
    }
}

// ---------------------------------------------------------------------------
// Operator registration
//   Two inputs: self (float16 tensor), other (float16 tensor)
//   One output: out (float16 tensor)
// ---------------------------------------------------------------------------
IMPLEMT_COMMON(GcdCustom, "gcd_custom",
               INPUTS({ { "self", GE_FLOAT16, TENSOR },
                        { "other", GE_FLOAT16, TENSOR } }),
               OUTPUTS({ { "out", GE_FLOAT16, TENSOR } }));

/**
 * Tiling function.
 *
 * Given two 4D input shapes [N4,N3,N2,N1], computes:
 *   - broadcast output shape
 *   - total output elements
 *   - strides for self, other (with 0 for broadcast dims)
 *   - strides for output (for flat index -> 4D index conversion)
 */
IMPLEMT_COMMON_TILING(GcdCustom, "gcd_custom") {
    auto op = TilingOp("gcd_custom", tilingData->blockDim, args->GetInput(0));
    auto &selfShape = args->GetInput(0)->GetShape();
    auto &otherShape = args->GetInput(1)->GetShape();

    // ---- Read input shapes (assume 4D) ----
    size_t dimNum = selfShape.GetDimNum();
    std::vector<int64_t> selfVec(dimNum);
    std::vector<int64_t> otherVec(dimNum);
    for (size_t i = 0; i < dimNum; i++) {
        selfVec[i] = selfShape.GetDim(i);
        otherVec[i] = otherShape.GetDim(i);
    }

    // Pad to 4D (if shapes have fewer dims, pad with 1 on the left)
    std::vector<int64_t> self4D(4, 1);
    std::vector<int64_t> other4D(4, 1);
    size_t offset = 4 - dimNum;
    for (size_t i = 0; i < dimNum; i++) {
        self4D[offset + i] = selfVec[i];
        other4D[offset + i] = otherVec[i];
    }

    // ---- Compute broadcast output shape ----
    int64_t outShape4D[4];
    BroadcastShape4D(self4D, other4D, outShape4D);

    // ---- Compute strides for broadcast mapping ----
    int32_t selfStrides[4];
    int32_t otherStrides[4];
    int32_t outStrides[4];
    ComputeBroadcastStrides(self4D.data(), selfStrides);
    ComputeBroadcastStrides(other4D.data(), otherStrides);
    ComputeBroadcastStrides(outShape4D, outStrides);

    // ---- Total output elements ----
    uint32_t totalLength = 1;
    for (int32_t d = 0; d < 4; d++) {
        totalLength *= (uint32_t)outShape4D[d];
    }

    // ---- Determine tile length ----
    uint32_t tileLen = std::min(totalLength, (uint32_t)2048u);
    tileLen = (tileLen / 16) * 16;
    if (tileLen == 0) tileLen = 16;

    // ---- Serialise tiling data ----
    GcdCustomTilingData tiling;
    tiling.totalLength = totalLength;
    tiling.tileLen = tileLen;
    for (int32_t d = 0; d < 4; d++) {
        tiling.selfStrides[d]  = selfStrides[d];
        tiling.otherStrides[d] = otherStrides[d];
        tiling.outStrides[d]   = outStrides[d];
    }
    tilingData->SetRawTiling(&tiling, sizeof(GcdCustomTilingData));

    // ---- Set block dimension ----
    uint32_t maxBlocks = 8;
    uint32_t blockDim = std::min(maxBlocks, (totalLength + tileLen - 1) / tileLen);
    if (blockDim == 0) blockDim = 1;
    tilingData->blockDim = blockDim;

    return TilingSuccess;
}

// ---------------------------------------------------------------------------
// Infer shape: output shape = broadcast(self.shape, other.shape)
// ---------------------------------------------------------------------------
INFER_SHAPE(GcdCustom, "gcd_custom") {
    auto &selfShape  = args->GetInput(0)->GetShape();
    auto &otherShape = args->GetInput(1)->GetShape();
    auto *outShape   = args->GetOutput(0)->MutShape();

    size_t dimNum = selfShape.GetDimNum();
    // Output shape takes the max of each dimension (broadcast)
    for (size_t i = 0; i < dimNum; i++) {
        int64_t dim = std::max(selfShape.GetDim(i), otherShape.GetDim(i));
        outShape->SetDim(i, dim);
    }
    outShape->SetDimNum(dimNum);

    return GraphSuccess;
}
