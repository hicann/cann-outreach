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
// kernel 侧 BUFFER_NUM=2（DoubleBuffer）：1 输入×2 + 1 输出×2 = 4
constexpr int64_t DOUBLE_BUF_TENSOR_COUNT = 4;

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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    int64_t totalIdx = inputShapeX.GetShapeSize();

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    int64_t typeSize = ge::GetSizeByDataType(dataType);
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "invalid dtype size"), return ge::GRAPH_FAILED);

    tiling->totalNum = totalIdx;
    if (totalIdx == 0) {
        tiling->blockFactor = 1;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
    } else {
        int64_t ubBlockSize = GetUbBlockSize(context);
        OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context, "ubBlockSize is 0"), return ge::GRAPH_FAILED);

        // 多核切分：按核均分并对齐到 32B 粒度，避免 CopyOut 踩相邻核
        tiling->blockFactor = CeilAlign(CeilDiv(totalIdx, coreNum), ubBlockSize);
        int64_t usedCoreNum = CeilDiv(totalIdx, tiling->blockFactor);

        // UB 切分：与 kernel DoubleBuffer 一致，预留 in/out 各 BUFFER_NUM 份
        tiling->ubFactor = FloorAlign(
            FloorDiv(static_cast<int64_t>(ubSize) / typeSize, DOUBLE_BUF_TENSOR_COUNT), ubBlockSize);
        OP_CHECK_IF(tiling->ubFactor <= 0, OP_LOGE(context, "ubFactor is 0"), return ge::GRAPH_FAILED);

        context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    }

    // 根据输入 dtype 选择 tilingKey：fp16/bf16 -> mode0，其余(float) -> mode1
    uint64_t tilingKey;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
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
