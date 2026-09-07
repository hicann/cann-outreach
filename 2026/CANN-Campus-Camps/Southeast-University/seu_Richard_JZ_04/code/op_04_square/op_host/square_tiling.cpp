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

static ge::graphStatus SquareTilingFunc(gert::TilingContext* context)
{
    // TODO: 实现 Tiling 逻辑
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

    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = inputShape->GetOriginShape().GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    int64_t typeSize = 0;
    uint64_t tilingKey = 0;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        typeSize = sizeof(uint16_t);
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0);
    } else if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        typeSize = sizeof(float);
        tilingKey = GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1);
    } else {
        OP_LOGE(context, "Square only supports float16 and float32");
        return ge::GRAPH_FAILED;
    }

    // 数据量较小时单核运行；较大时按约 1024 个元素/核启用多核，最多使用全部 AIV 核。
    int64_t blockDim = 1;
    if (totalNum >= MIN_SPLIT_THRESHOLD) {
        blockDim = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
        if (blockDim > coreNum) {
            blockDim = coreNum;
        }
    }

    // 每核处理连续的一段数据。最后一个核在 Kernel Init 中用 totalNum 截断。
    const int64_t blockFactor = CeilDiv(totalNum, blockDim);

    // Kernel 有 input/output 两个队列，每个队列 BUFFER_NUM=2，故 UB 中共需 4 个 buffer。
    // 每个 buffer 的字节数向下取 32B 对齐，保证不会因 InitBuffer 向上补齐而超 UB。
    constexpr int64_t UB_BUFFER_COUNT = 4;
    constexpr int64_t DATA_BLOCK_BYTES = 32;
    int64_t perBufferBytes = static_cast<int64_t>(ubSize) / UB_BUFFER_COUNT;
    perBufferBytes = (perBufferBytes / DATA_BLOCK_BYTES) * DATA_BLOCK_BYTES;
    OP_CHECK_IF(perBufferBytes < typeSize, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);

    int64_t ubFactor = perBufferBytes / typeSize;
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubFactor is invalid"), return ge::GRAPH_FAILED);

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
