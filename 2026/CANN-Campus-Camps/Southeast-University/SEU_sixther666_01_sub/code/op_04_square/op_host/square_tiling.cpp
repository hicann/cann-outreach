/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TARGET_ELEMENTS_PER_CORE = 1024;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t UB_TENSOR_NUM = 2 * DOUBLE_BUFFER_NUM; // 1 input + 1 output, both double buffered
constexpr int64_t VECTOR_BYTES = 256;
constexpr int64_t MAX_REPEAT_TIMES = 255;

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
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum must be positive"), return ge::GRAPH_FAILED);
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

static ge::graphStatus SquareTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    SquareTilingData* tiling = context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        OP_CHECK_IF(storageShape.GetDim(i) <= 0,
                    OP_LOGE(context, "input dimensions must be positive, but dim %zu is %ld", i,
                            storageShape.GetDim(i)),
                    return ge::GRAPH_FAILED);
    }
    const int64_t totalNum = storageShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "input shape size must be positive, but got %ld", totalNum),
                return ge::GRAPH_FAILED);

    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    const gert::Shape outputStorageShape = EnsureNotScalar(outputShape->GetStorageShape());
    OP_CHECK_IF(outputStorageShape.GetDimNum() != storageShape.GetDimNum(),
                OP_LOGE(context, "output rank must be the same as input rank"), return ge::GRAPH_FAILED);
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        OP_CHECK_IF(outputStorageShape.GetDim(i) != storageShape.GetDim(i),
                    OP_LOGE(context, "output shape must be the same as input shape"), return ge::GRAPH_FAILED);
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    int64_t typeSize = 0;
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        typeSize = static_cast<int64_t>(sizeof(uint16_t));
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0);
    } else if (dataType == ge::DT_FLOAT) {
        typeSize = static_cast<int64_t>(sizeof(float));
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "Square only supports float16 and float32");
        return ge::GRAPH_FAILED;
    }
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);
    OP_CHECK_IF(outputDesc->GetDataType() != dataType,
                OP_LOGE(context, "output dtype must be the same as input dtype"), return ge::GRAPH_FAILED);

    const int64_t ubBlockSize = GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0 || ubBlockSize % typeSize != 0,
                OP_LOGE(context, "invalid UB block size: %ld", ubBlockSize), return ge::GRAPH_FAILED);
    const int64_t alignNum = ubBlockSize / typeSize;

    // Avoid launching many cores for only a few bytes of work. The selected core
    // count keeps roughly 1024 elements on each core until all AIV cores are used.
    const int64_t targetCoreNum = CeilDiv(totalNum, TARGET_ELEMENTS_PER_CORE);
    const int64_t splitCoreNum = targetCoreNum < coreNum ? targetCoreNum : coreNum;
    int64_t blockFactor = totalNum;
    if (splitCoreNum > 1) {
        blockFactor = CeilAlign(CeilDiv(totalNum, splitCoreNum), alignNum);
    }
    const int64_t usedCoreNum = CeilDiv(totalNum, blockFactor);

    // Four UB buffers are live with double buffering. Cap one tile to the
    // calCount range of the vector Mul API (at most 255 repeats).
    const int64_t ubCapacity = FloorDiv(static_cast<int64_t>(ubSize), typeSize * UB_TENSOR_NUM);
    const int64_t maxVectorElements = FloorDiv(MAX_REPEAT_TIMES * VECTOR_BYTES, typeSize);
    const int64_t maxUbFactor = FloorAlign(ubCapacity < maxVectorElements ? ubCapacity : maxVectorElements, alignNum);
    const int64_t alignedBlockFactor = CeilAlign(blockFactor, alignNum);
    const int64_t ubFactor = alignedBlockFactor < maxUbFactor ? alignedBlockFactor : maxUbFactor;
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "UB is too small for Square"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square).Tiling(SquareTilingFunc).TilingParse<SquareCompileInfo>(TilingParseForSquare);

} // namespace optiling
