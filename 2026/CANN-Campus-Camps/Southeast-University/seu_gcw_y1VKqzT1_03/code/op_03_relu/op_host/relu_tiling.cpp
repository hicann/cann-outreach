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
// 1 路输入 + 1 路输出，且 kernel 中均使用 DoubleBuffer。
constexpr int64_t UB_BUFFER_NUM = 4;
// DataCopyPad 的单次搬运长度保持在一个较保守的范围内，同时避免占满全部 UB。
constexpr int64_t MAX_UB_FACTOR = 8192;

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

    // 1. 获取输入 shape 和 dtype。
    const gert::StorageShape* inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    const gert::Shape inputShape = EnsureNotScalar(inputX->GetStorageShape());
    const int64_t totalNum = inputShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "invalid input size: %ld", totalNum), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "unsupported dtype: %d", static_cast<int32_t>(dataType)),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 2. Core 级切分。
    // 每个核尽量至少处理 MIN_SPLIT_THRESHOLD 个元素，避免小张量过度拆核。
    int64_t usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    usedCoreNum = (usedCoreNum < 1) ? 1 : usedCoreNum;
    usedCoreNum = (usedCoreNum > coreNum) ? coreNum : usedCoreNum;

    tiling->totalNum = totalNum;
    tiling->blockFactor = CeilDiv(totalNum, usedCoreNum);

    // 3. UB 级切分。
    // 以 float32 的 4B 作为保守上界计算 UB 容量，因此对 float16 同样安全。
    // kernel 里有 input/output 两个队列且均为双缓冲，共需要 4 份 tile 空间。
    const int64_t ubBlockSize = GetUbBlockSize(context);
    OP_CHECK_IF(ubBlockSize <= 0, OP_LOGE(context, "invalid ubBlockSize"), return ge::GRAPH_FAILED);

    int64_t ubFactor = FloorAlign(FloorDiv(static_cast<int64_t>(ubSize) / TYPE_SIZE, UB_BUFFER_NUM), ubBlockSize);
    if (ubFactor > MAX_UB_FACTOR) {
        ubFactor = FloorAlign(MAX_UB_FACTOR, ubBlockSize);
    }
    OP_CHECK_IF(ubFactor <= 0, OP_LOGE(context, "ubFactor is 0"), return ge::GRAPH_FAILED);
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    // 4. dtype 分支：mode 0 -> float16，mode 1 -> float32。
    uint64_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
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
