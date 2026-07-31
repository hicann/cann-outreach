/*!
 * \file soft_shrink_grad_tiling.cpp
 * \brief SoftShrinkGrad 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/soft_shrink_grad_tiling_data.h"
#include "../op_kernel/soft_shrink_grad_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t FLOAT_SIZE = 4;
constexpr int64_t HALF_SIZE = 2;
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t SELECT_TMP_RESERVED = 8 * 1024;

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

static int64_t GetShapeSize(const gert::Shape& shape)
{
    return shape.GetDimNum() == 0 ? 1 : shape.GetShapeSize();
}

static ge::graphStatus SoftShrinkGradTilingFunc(gert::TilingContext* context)
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

    SoftShrinkGradTilingData* tiling = context->GetTilingData<SoftShrinkGradTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    int64_t totalNum = GetShapeSize(inputShape->GetStorageShape());
    tiling->totalNum = totalNum;
    tiling->blockFactor = totalNum > 0 ? CeilDiv(totalNum, coreNum) : 1;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    int64_t typeSize = inputDesc->GetDataType() == ge::DT_FLOAT ? FLOAT_SIZE : HALF_SIZE;
    int64_t calcBytes = 0;
    if (inputDesc->GetDataType() == ge::DT_FLOAT16) {
        calcBytes = FLOAT_SIZE; // fp32 copy of x for threshold comparisons.
    } else if (inputDesc->GetDataType() == ge::DT_BF16) {
        calcBytes = FLOAT_SIZE * 3; // x, grad and result in fp32.
    }
    int64_t usableUb = static_cast<int64_t>(ubSize) - SELECT_TMP_RESERVED;
    OP_CHECK_IF(usableUb <= 0, OP_LOGE(context, "UB is too small for SoftShrinkGrad"), return ge::GRAPH_FAILED);
    int64_t bytesPerElement = typeSize * BUFFER_NUM * 3 + calcBytes + 1; // +1 amortizes the compare mask.
    int64_t ubFactor = FloorAlign(usableUb / bytesPerElement, static_cast<int64_t>(64));
    tiling->ubFactor = ubFactor > 0 ? ubFactor : 64;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const float* lambd = attrs->GetAttrPointer<float>(0);
        if (lambd != nullptr) {
            tiling->lambd = *lambd;
        }
    }

    context->SetBlockDim(totalNum > 0 ? CeilDiv(totalNum, tiling->blockFactor) : 1);

    uint64_t tilingKey;
    switch (inputDesc->GetDataType()) {
        case ge::DT_FLOAT16:
            tilingKey = GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_0);
            break;
        case ge::DT_FLOAT:
            tilingKey = GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_1);
            break;
        case ge::DT_BF16:
            tilingKey = GET_TPL_TILING_KEY(SOFTSHRINKGRAD_TPL_SCH_MODE_2);
            break;
        default:
            OP_LOGE(context, "Unsupported dtype for SoftShrinkGrad");
            return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSoftShrinkGrad([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SoftShrinkGradCompileInfo {};

IMPL_OP_OPTILING(SoftShrinkGrad).Tiling(SoftShrinkGradTilingFunc).TilingParse<SoftShrinkGradCompileInfo>(TilingParseForSoftShrinkGrad);

} // namespace optiling
