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
constexpr int64_t MIN_PER_CORE_FP32 = 2048;
constexpr int64_t MIN_PER_CORE_FP16 = 4096;

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

static inline int64_t GetDtypeSize(ge::DataType dataType)
{
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        return 2;
    }
    return 4;
}

static ge::graphStatus GetShapeAttrsInfo(gert::TilingContext* context, int64_t& totalNum, int64_t& typeSize)
{
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    auto outY = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outY);
    auto outShapeY = EnsureNotScalar(outY->GetStorageShape());

    int64_t inputShapeSize = static_cast<int64_t>(inputShapeX.GetShapeSize());
    int64_t outputShapeSize = static_cast<int64_t>(outShapeY.GetShapeSize());
    OP_CHECK_IF(inputShapeSize != outputShapeSize,
        OP_LOGE(context, "Relu: shape mismatch: x=%ld, y=%ld", inputShapeSize, outputShapeSize),
        return ge::GRAPH_FAILED);

    totalNum = inputShapeSize;
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    if (dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16 && dataType != ge::DT_BF16) {
        OP_LOGE(context, "Relu: unsupported dtype");
        return ge::GRAPH_FAILED;
    }
    typeSize = GetDtypeSize(dataType);
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

    int64_t totalNum;
    int64_t typeSize;
    OP_CHECK_IF(
        GetShapeAttrsInfo(context, totalNum, typeSize) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetShapeAttrsInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    if (totalNum <= 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 1;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // fp16 核数更少（降调度开销），fp32 核数更多（提带宽），均按每核最小元素数切分
    int64_t ubBlockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    int64_t minPerCore = (typeSize == 2) ? MIN_PER_CORE_FP16 : MIN_PER_CORE_FP32;
    int64_t usedCoreNum = CeilDiv(totalNum, minPerCore);
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    tiling->blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), ubBlockSize);
    usedCoreNum = CeilDiv(totalNum, tiling->blockFactor);

    // 双缓冲：输入/输出各 2 块 buffer，共占用 4 份单缓冲
    constexpr int64_t BUFFER_TOTAL = 2 * 2;
    tiling->ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / typeSize, BUFFER_TOTAL), ubBlockSize);
    if (tiling->ubFactor <= 0) {
        OP_LOGE(context, "ubFactor is 0");
        return ge::GRAPH_FAILED;
    }

    tiling->totalNum = totalNum;
    context->SetBlockDim(usedCoreNum);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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