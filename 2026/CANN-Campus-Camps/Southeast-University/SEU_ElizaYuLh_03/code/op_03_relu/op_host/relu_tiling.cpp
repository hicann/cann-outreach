/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape& shape = xShape->GetStorageShape();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    int64_t typeSize = TYPE_SIZE;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        typeSize = 2;
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        typeSize = 4;
    } else {
        OP_LOGE(context, "Relu only supports FLOAT16/FLOAT, dtype=%d",
                static_cast<int>(inputDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }

    int64_t totalNum = static_cast<int64_t>(shape.GetShapeSize());
    if (totalNum < 0) {
        OP_LOGE(context, "Invalid input element count");
        return ge::GRAPH_FAILED;
    }

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 每个AI Core至少处理MIN_SPLIT_THRESHOLD个元素，避免小输入启动过多核。
    int64_t blockDim = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (blockDim < 1) {
        blockDim = 1;
    }
    if (blockDim > coreNum) {
        blockDim = coreNum;
    }

    // DataCopy要求数据地址满足32B对齐；blockFactor按元素数做32B对齐。
    const int64_t blockAlign = 32 / typeSize;
    int64_t blockFactor = CeilDiv(totalNum, blockDim);
    blockFactor = CeilAlign(blockFactor, blockAlign);

    // inputQueueX + outputQueueY，各开启2个buffer。
    // 因而一个tile最多占用 4 * ubFactor * sizeof(T) 字节。
    const uint64_t queueNum = 4;
    uint64_t ubFactor = ubSize / (queueNum * static_cast<uint64_t>(typeSize));
    ubFactor = (ubFactor / static_cast<uint64_t>(blockAlign)) * static_cast<uint64_t>(blockAlign);
    if (ubFactor == 0) {
        OP_LOGE(context, "UB is too small for Relu tile");
        return ge::GRAPH_FAILED;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = static_cast<int64_t>(ubFactor);

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    uint64_t tilingKey;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
