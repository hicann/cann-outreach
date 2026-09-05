/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
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
constexpr uint32_t BUFFER_NUM = 2U;
constexpr uint32_t QUEUE_NUM = 2U;

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

static ge::graphStatus SquareTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    OP_CHECK_IF(GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize error"), return ge::GRAPH_FAILED);

    const gert::StorageShape* storageShape = context->GetInputShape(0);
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, storageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const gert::Shape inputShape = EnsureNotScalar(storageShape->GetOriginShape());
    const int64_t totalNum = inputShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "input element count must be positive"), return ge::GRAPH_FAILED);

    uint32_t typeSize = 0;
    uint64_t tilingKey = 0;
    switch (inputDesc->GetDataType()) {
        case ge::DT_FLOAT16:
            typeSize = sizeof(uint16_t);
            tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0);
            break;
        case ge::DT_FLOAT:
            typeSize = sizeof(float);
            tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1);
            break;
        default:
            OP_LOGE(context, "Square only supports float16 and float32");
            return ge::GRAPH_FAILED;
    }

    const uint64_t ubBlockSize = GetUbBlockSize(context);
    // 输入队列和输出队列均为双缓冲，因此 UB 被划分为 2 * 2 个同等大小的 buffer。
    const uint64_t ubBytesPerBuffer =
        (ubSize / (QUEUE_NUM * BUFFER_NUM) / ubBlockSize) * ubBlockSize;
    OP_CHECK_IF(ubBytesPerBuffer < ubBlockSize,
                OP_LOGE(context, "UB size is too small for Square buffers"), return ge::GRAPH_FAILED);

    const int64_t ubFactor = static_cast<int64_t>(ubBytesPerBuffer / typeSize);
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "UB tile length is zero"), return ge::GRAPH_FAILED);

    const int64_t blockDim = (totalNum < coreNum) ? totalNum : coreNum;
    const int64_t blockFactor = CeilDiv(totalNum, blockDim);

    SquareTilingData* tiling = context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));
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
