/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file truncate_div_tiling.cpp
 * \brief tiling implementation for TruncateDiv
 *
 * Responsibilities:
 *   - Platform info retrieval (core count, UB size)
 *   - Input validation (dtype, format, broadcast legality)
 *   - Broadcast output shape derivation & mode classification
 *   - Core split (output-linear space) & UB tile sizing
 *   - Stride table generation for kernel-side address calculation
 *   - TilingKey encoding and blockDim setting
 *   - Workspace size reporting
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <vector>
#include <algorithm>

#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/truncate_div_tiling_data.h"

namespace optiling {

using namespace platform_ascendc;

// ---------------------------------------------------------------------------
//  Constants
// ---------------------------------------------------------------------------
static constexpr uint32_t ALIGN_BYTES      = 256;   // 32B-aligned × 8 （UB burst alignment unit）
static constexpr uint32_t UB_RESERVE_BYTES = 4096;  // reserved UB for stack / framework overhead

static constexpr uint32_t MIN_ELEMS_PER_CORE = 2048; // avoid excessive core count on small shapes

// ---------------------------------------------------------------------------
//  Supported data types
// ---------------------------------------------------------------------------
static const std::set<ge::DataType> supportedDtype = {
    ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16,
    ge::DT_INT8,    ge::DT_UINT8, ge::DT_INT32
};

// ---------------------------------------------------------------------------
//  Helper: map ge::DataType -> dtypeMode
// ---------------------------------------------------------------------------
static uint32_t DtypeToMode(ge::DataType dt) {
    switch (dt) {
        case ge::DT_FLOAT16: return 0;
        case ge::DT_FLOAT:   return 1;
        case ge::DT_BF16:    return 2;
        case ge::DT_INT8:    return 3;
        case ge::DT_UINT8:   return 4;
        case ge::DT_INT32:   return 5;
        default:             return 0xFFFFFFFFu;
    }
}

// ---------------------------------------------------------------------------
//  Helper: size of one element in bytes
// ---------------------------------------------------------------------------
static uint32_t ElemBytes(ge::DataType dt) {
    switch (dt) {
        case ge::DT_FLOAT16: return 2;
        case ge::DT_FLOAT:   return 4;
        case ge::DT_BF16:    return 2;
        case ge::DT_INT8:    return 1;
        case ge::DT_UINT8:   return 1;
        case ge::DT_INT32:   return 4;
        default:             return 4;
    }
}

// ---------------------------------------------------------------------------
//  Helper: ceil division avoiding overflow
// ---------------------------------------------------------------------------
static inline uint64_t CeilDiv(uint64_t a, uint64_t b) {
    return (a + b - 1) / b;
}

static inline uint64_t AlignUp(uint64_t val, uint64_t align) {
    return (val + align - 1) / align * align;
}

// ---------------------------------------------------------------------------
//  1. Platform info
// ---------------------------------------------------------------------------
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context,
                                       int64_t& coreNum, uint32_t& wsSysSize) {
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0,
                OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    wsSysSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
//  2. UB size
// ---------------------------------------------------------------------------
static ge::graphStatus GetUBSize(gert::TilingContext* context, uint64_t& ubSize) {
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= 0,
                OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
//  3. Shape / attr info gathering
// ---------------------------------------------------------------------------
struct TruncateDivInputInfo {
    ge::DataType dataType;
    uint64_t totalLength;        // output total elements
    uint32_t elemBytes;
    uint32_t dtypeMode;

    // Standardised right-aligned shapes (padded to TRUNCATE_DIV_MAX_DIM)
    uint32_t rank;
    std::vector<uint64_t> x1Shape;
    std::vector<uint64_t> x2Shape;
    std::vector<uint64_t> outShape;

    uint32_t broadcastMode;      // 0-4
};

static ge::graphStatus GetShapeInfo(gert::TilingContext* context,
                                    TruncateDivInputInfo& info) {
    // --- Input shapes ---
    const gert::Shape* x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape* x2Shape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Shape);

    auto x1Storage = x1Shape->GetStorageShape();
    auto x2Storage = x2Shape->GetStorageShape();
    uint32_t rank1 = x1Storage.GetDimNum();
    uint32_t rank2 = x2Storage.GetDimNum();
    info.rank = (rank1 > rank2) ? rank1 : rank2;
    if (info.rank > TRUNCATE_DIV_MAX_DIM) {
        OP_LOGE(context, "rank %u exceeds max %u", info.rank, TRUNCATE_DIV_MAX_DIM);
        return ge::GRAPH_FAILED;
    }

    // Right-align and standardise shapes
    info.x1Shape.assign(TRUNCATE_DIV_MAX_DIM, 1);
    info.x2Shape.assign(TRUNCATE_DIV_MAX_DIM, 1);
    info.outShape.assign(TRUNCATE_DIV_MAX_DIM, 1);

    for (uint32_t i = 0; i < rank1; ++i) {
        int64_t d = x1Storage.GetDim(i);
        if (d < 0) {
            OP_LOGE(context, "x1 has unresolved dimension %lld at axis %u", d, i);
            return ge::GRAPH_FAILED;
        }
        info.x1Shape[TRUNCATE_DIV_MAX_DIM - rank1 + i] = static_cast<uint64_t>(d);
    }
    for (uint32_t i = 0; i < rank2; ++i) {
        int64_t d = x2Storage.GetDim(i);
        if (d < 0) {
            OP_LOGE(context, "x2 has unresolved dimension %lld at axis %u", d, i);
            return ge::GRAPH_FAILED;
        }
        info.x2Shape[TRUNCATE_DIV_MAX_DIM - rank2 + i] = static_cast<uint64_t>(d);
    }

    // Broadcast derivation
    uint64_t total = 1;
    for (uint32_t i = 0; i < TRUNCATE_DIV_MAX_DIM; ++i) {
        uint64_t d1 = info.x1Shape[i];
        uint64_t d2 = info.x2Shape[i];
        if (d1 != d2 && d1 != 1 && d2 != 1) {
            OP_LOGE(context, "broadcast fail at axis %u: x1=%llu x2=%llu",
                    i, d1, d2);
            return ge::GRAPH_FAILED;
        }
        uint64_t od = (d1 == d2) ? d1 : ((d1 == 1) ? d2 : d1);
        info.outShape[i] = od;
        if (od != 0 && total > std::numeric_limits<uint64_t>::max() / od) {
            OP_LOGE(context, "output element count overflows uint64 at axis %u", i);
            return ge::GRAPH_FAILED;
        }
        total *= od;
    }
    info.totalLength = total;

    // --- Broadcast mode classification ---
    bool x1Scalar = (rank1 == 0) || (x1Storage.GetDimNum() == 1 && x1Storage.GetDim(0) == 1);
    bool x2Scalar = (rank2 == 0) || (x2Storage.GetDimNum() == 1 && x2Storage.GetDim(0) == 1);

    if (x1Storage == x2Storage && !x1Scalar) {
        info.broadcastMode = 0;  // same shape
    } else if (x1Scalar && !x2Scalar) {
        info.broadcastMode = 1;  // x1 scalar
    } else if (!x1Scalar && x2Scalar) {
        info.broadcastMode = 2;  // x2 scalar
    } else {
        // Classify broadcast mode 3 (unilateral) vs 4 (general).
        // Unilateral: for every axis where the two shapes differ,
        //             exactly one input has a 1 (the other matches output).
        // General:    at least one axis where both inputs differ from 1
        //             AND from each other.
        bool x1HasExtra = false;
        bool x2HasExtra = false;
        info.broadcastMode = 3;  // default: unilateral
        for (uint32_t i = 0; i < TRUNCATE_DIV_MAX_DIM; ++i) {
            if (info.x1Shape[i] != info.x2Shape[i]) {
                if (info.x1Shape[i] == 1) {
                    x2HasExtra = true;   // x2 has a non-1 where x1 is 1
                } else if (info.x2Shape[i] == 1) {
                    x1HasExtra = true;   // x1 has a non-1 where x2 is 1
                } else {
                    // Neither is 1 → general broadcast
                    info.broadcastMode = 4;
                    break;
                }
            }
        }
        // If both sides have at least one non-1 expansion → general
        if (info.broadcastMode == 3 && x1HasExtra && x2HasExtra) {
            info.broadcastMode = 4;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
//  4. Dtype info
// ---------------------------------------------------------------------------
static ge::graphStatus GetDtypeInfo(gert::TilingContext* context,
                                    TruncateDivInputInfo& info) {
    const gert::TensorDesc* x1Desc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Desc);
    const gert::TensorDesc* x2Desc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2Desc);

    ge::DataType dt = x1Desc->GetDataType();
    if (x2Desc->GetDataType() != dt) {
        OP_LOGE(context, "input dtype mismatch");
        return ge::GRAPH_FAILED;
    }
    if (supportedDtype.count(dt) == 0) {
        OP_LOGE(context, "unsupported dtype %d", static_cast<int>(dt));
        return ge::GRAPH_FAILED;
    }

    info.dataType = dt;
    info.elemBytes = ElemBytes(dt);
    info.dtypeMode = DtypeToMode(dt);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
//  5. Workspace size
// ---------------------------------------------------------------------------
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context,
                                        uint32_t wsSysSize) {
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = wsSysSize;  // only system workspace needed
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
//  6. Stride table generation
// ---------------------------------------------------------------------------
static void ComputeStrides(const std::vector<uint64_t>& shape,
                           uint64_t strides[TRUNCATE_DIV_MAX_DIM]) {
    uint64_t acc = 1;
    for (int32_t i = static_cast<int32_t>(TRUNCATE_DIV_MAX_DIM) - 1; i >= 0; --i) {
        if (shape[i] == 1) {
            strides[i] = 0;  // broadcast dim: stride = 0
        } else {
            strides[i] = acc;
            acc *= shape[i];
        }
    }
}

// ---------------------------------------------------------------------------
//  7. Main tiling function
// ---------------------------------------------------------------------------
static ge::graphStatus TruncateDivTilingFunc(gert::TilingContext* context) {
    // ---- Step 1: platform info ----
    int64_t coreNum;
    uint32_t wsSysSize;
    OP_CHECK_IF(GetPlatformInfo(context, coreNum, wsSysSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"), return ge::GRAPH_FAILED);

    uint64_t ubSize;
    OP_CHECK_IF(GetUBSize(context, ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetUBSize failed"), return ge::GRAPH_FAILED);

    // ---- Step 2: input shape & dtype info ----
    TruncateDivInputInfo info;
    OP_CHECK_IF(GetDtypeInfo(context, info) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetDtypeInfo failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeInfo(context, info) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetShapeInfo failed"), return ge::GRAPH_FAILED);

    // ---- Step 3: workspace ----
    OP_CHECK_IF(GetWorkspaceSize(context, wsSysSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize failed"), return ge::GRAPH_FAILED);

    // ---- Step 4: tiling data ----
    TruncateDivTilingData* tiling = context->GetTilingData<TruncateDivTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(TruncateDivTilingData), 0,
                         sizeof(TruncateDivTilingData)) != EOK,
                OP_LOGE(context, "memset tiling failed"), return ge::GRAPH_FAILED);

    // ---- Step 5: UB tile sizing ----
    // Per-element UB consumption (double buffer: 2 slots each for x1, x2, y)
    uint32_t ubQueueBytesPerElem = 2 * (info.elemBytes * 3);  // x1 + x2 + y
    // Float32 intermediate buffers for computation
    uint32_t ubCalcBytesPerElem = 4 * sizeof(float);           // x1_fp32, x2_fp32, y_fp32, tmp for trunc
    // Extra for int8/uint8 type promotion to half
    uint32_t ubCastBytesPerElem = (info.dtypeMode == 3 || info.dtypeMode == 4)
                                      ? sizeof(uint16_t) * 2   // half intermediate for x1, x2
                                      : 0;
    uint32_t perElemUb = ubQueueBytesPerElem + ubCalcBytesPerElem + ubCastBytesPerElem;
    uint64_t usableUb = (ubSize > UB_RESERVE_BYTES) ? (ubSize - UB_RESERVE_BYTES) : ubSize;

    // ALIGN_BYTES alignment for DataCopy burst
    uint32_t alignElems = ALIGN_BYTES / info.elemBytes;
    if (alignElems == 0) alignElems = 1;

    uint64_t tileLength = (usableUb / perElemUb / alignElems) * alignElems;
    if (tileLength < alignElems) tileLength = alignElems;  // at least one aligned block

    // ---- Step 6: core split ----
    uint64_t totalLength = info.totalLength;
    uint64_t coreLength;
    uint32_t usedCoreNum;

    if (totalLength <= MIN_ELEMS_PER_CORE) {
        // Very small shape: single core
        usedCoreNum = 1;
        coreLength = totalLength;
    } else {
        // Estimate initial core count from total workload
        uint32_t targetCore = static_cast<uint32_t>(
            CeilDiv(totalLength * info.elemBytes * 3, 16 * 1024));  // ~16 KiB per core
        targetCore = std::min(targetCore, static_cast<uint32_t>(coreNum));
        targetCore = std::max(targetCore, 1u);

        // Align per-core length
        coreLength = AlignUp(CeilDiv(totalLength, targetCore), alignElems);
        usedCoreNum = static_cast<uint32_t>(CeilDiv(totalLength, coreLength));
        if (usedCoreNum > static_cast<uint32_t>(coreNum)) {
            usedCoreNum = static_cast<uint32_t>(coreNum);
            coreLength = AlignUp(CeilDiv(totalLength, usedCoreNum), alignElems);
        }
    }

    // ---- Step 7: stride tables ----
    uint64_t x1Stride[TRUNCATE_DIV_MAX_DIM];
    uint64_t x2Stride[TRUNCATE_DIV_MAX_DIM];
    ComputeStrides(info.x1Shape, x1Stride);
    ComputeStrides(info.x2Shape, x2Stride);

    // ---- Step 8: populate tiling data ----
    tiling->totalLength     = totalLength;
    tiling->coreLength      = coreLength;
    tiling->tileLength      = static_cast<uint32_t>(tileLength);
    tiling->usedCoreNum     = usedCoreNum;
    tiling->dtypeMode       = info.dtypeMode;
    tiling->broadcastMode   = info.broadcastMode;

    for (uint32_t i = 0; i < TRUNCATE_DIV_MAX_DIM; ++i) {
        tiling->outputShape[i] = info.outShape[i];
        tiling->x1Stride[i]    = x1Stride[i];
        tiling->x2Stride[i]    = x2Stride[i];
    }

    // ---- Step 9: blockDim & tilingKey ----
    context->SetBlockDim(usedCoreNum);

    // The current kernel dispatches from fields in TilingData at runtime.
    // Keep a single build variant until compile-time templates are introduced.
    uint64_t tilingKey = 0;
    context->SetTilingKey(tilingKey);

    OP_LOGD(context->GetNodeName(),
            "TruncateDiv tiling done: total=%llu, coreLen=%llu, tileLen=%u, "
            "cores=%u, dtypeMode=%u, bcastMode=%u, key=%llu",
            totalLength, coreLength,
            static_cast<uint32_t>(tileLength),
            usedCoreNum, info.dtypeMode, info.broadcastMode, tilingKey);

    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
//  Tiling parse (no-op for this operator)
// ---------------------------------------------------------------------------
static ge::graphStatus TilingParseForTruncateDiv(
    [[maybe_unused]] gert::TilingParseContext* context) {
    OP_CHECK_IF(context == nullptr,
                OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
//  Registration
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(TruncateDiv)
    .Tiling(TruncateDivTilingFunc)
    .TilingParse<TruncateDivTilingData>(TilingParseForTruncateDiv);

}  // namespace optiling
