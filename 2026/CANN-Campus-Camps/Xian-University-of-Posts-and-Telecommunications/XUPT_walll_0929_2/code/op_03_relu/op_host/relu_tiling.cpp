/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {


static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    const auto *inputDesc = context->GetInputDesc(0);
    const auto *inputShape = context->GetInputShape(0);
    if (inputDesc == nullptr || inputShape == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    const auto &shape = inputShape->GetStorageShape();
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        const int64_t dim = shape.GetDim(i);
        if (dim < 0 || (dim != 0 && totalNum > INT64_MAX / dim)) {
            return ge::GRAPH_FAILED;
        }
        totalNum *= dim;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum == 0 || ubSize < 128) {
        return ge::GRAPH_FAILED;
    }

    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : 4;
    const int64_t alignElems = 32 / typeSize;
    const int64_t tiles = totalNum / 1024 + (totalNum % 1024 != 0);
    int64_t blockDim = coreNum < 8 ? coreNum : 8;
    if (tiles < blockDim) {
        blockDim = tiles > 0 ? tiles : 1;
    }
    const int64_t perCore = totalNum / blockDim + (totalNum % blockDim != 0);
    if (perCore > INT64_MAX - alignElems + 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t blockFactor = perCore == 0 ? alignElems :
        ((perCore + alignElems - 1) / alignElems) * alignElems;
    blockDim = totalNum == 0 ? 1 :
        totalNum / blockFactor + (totalNum % blockFactor != 0);

    // Two input buffers and two output buffers share the UB budget.
    int64_t ubFactor = static_cast<int64_t>(ubSize / 4 / typeSize / alignElems) * alignElems;
    if (ubFactor > 1024) {
        ubFactor = 1024;
    }
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    auto *tiling = context->GetTilingData<ReluTilingData>();
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->GetRawTilingData()->SetDataSize(sizeof(ReluTilingData));
    context->SetBlockDim(static_cast<uint32_t>(blockDim));
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
