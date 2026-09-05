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
using Ops::Base::FloorAlign;
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MAX_LAST_DIM = 10240;
constexpr int64_t UB_TENSOR_NUM = 4; // 1 input + 1 output，均开启 double buffer

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
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
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetShapeAndType(gert::TilingContext* context, int64_t& totalNum, ge::DataType& dataType)
{
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const gert::Shape shape = EnsureNotScalar(inputShape->GetStorageShape());
    OP_CHECK_IF(shape.GetDimNum() == 0, OP_LOGE(context, "invalid scalar shape"), return ge::GRAPH_FAILED);

    const int64_t lastDim = shape.GetDim(shape.GetDimNum() - 1);
    OP_CHECK_IF(
        lastDim < 1 || lastDim > MAX_LAST_DIM,
        OP_LOGE(context, "last dimension should be in [1, 10240], but got %ld", lastDim),
        return ge::GRAPH_FAILED);

    totalNum = shape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "total element number should be positive"), return ge::GRAPH_FAILED);

    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "Square only supports float16 and float32"),
        return ge::GRAPH_FAILED);

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
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    int64_t totalNum = 0;
    ge::DataType dataType = ge::DT_FLOAT;
    OP_CHECK_IF(
        GetShapeAndType(context, totalNum, dataType) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAndType failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);

    ::SquareTilingData* tiling = context->GetTilingData<::SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 核间切分：优先尽量使用更多 AIV 核，尾核通过 kernel 侧 remainderLength 处理。
    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, coreNum);
    const int64_t usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // 核内切分：1 个输入 + 1 个输出，均为双缓冲，共占用 4 份 UB。
    // ubFactor 以元素数表示，并向下对齐到一个 UB data block 对应的元素数，
    // 从而保证 LocalTensor buffer 大小及向上补齐后的矢量计算都不会越界。
    const int64_t typeSize = (dataType == ge::DT_FLOAT) ? 4 : 2;
    const int64_t ubBlockBytes = GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockBytes <= 0, OP_LOGE(context, "invalid UB block size"), return ge::GRAPH_FAILED);

    const int64_t alignElements = ubBlockBytes / typeSize;
    OP_CHECK_IF(alignElements <= 0, OP_LOGE(context, "invalid alignment"), return ge::GRAPH_FAILED);

    const int64_t maxElementsPerBuffer = FloorDiv(static_cast<int64_t>(ubSize), typeSize * UB_TENSOR_NUM);
    tiling->ubFactor = FloorAlign(maxElementsPerBuffer, alignElements);
    OP_CHECK_IF(tiling->ubFactor <= 0, OP_LOGE(context, "UB is too small"), return ge::GRAPH_FAILED);

    context->SetBlockDim(usedCoreNum);

    // schMode=0 -> half，schMode=1 -> float，与 square.cpp 中模板分支保持一致。
    const uint32_t schMode = (dataType == ge::DT_FLOAT16) ? SQUARE_TPL_SCH_MODE_0 : SQUARE_TPL_SCH_MODE_1;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square).Tiling(SquareTilingFunc).TilingParse<SquareCompileInfo>(TilingParseForSquare);

} // namespace optiling
