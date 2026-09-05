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
constexpr int64_t UB_BUFFER_NUM = 4;
static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // 2. 获取输入 shape
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);

    const gert::Shape inputShape =
        EnsureNotScalar(inputShapePtr->GetStorageShape());

    const int64_t totalNum = inputShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum < 0,
        OP_LOGE(context, "invalid totalNum: %ld", totalNum),
        return ge::GRAPH_FAILED);

    // 3. 获取并检查输入 dtype
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const ge::DataType dataType = inputDesc->GetDataType();

    OP_CHECK_IF(
        dataType != ge::DT_FLOAT &&
            dataType != ge::DT_FLOAT16 &&
            dataType != ge::DT_BF16,
        OP_LOGE(context, "unsupported input dtype"),
        return ge::GRAPH_FAILED);

    // 4. 设置 workspace
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 5. 获取 TilingData
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;

    // UB Block 大小，用于保证切分结果满足硬件对齐要求
    const int64_t ubBlockSize =
        static_cast<int64_t>(GetUbBlockSize(context));

    OP_CHECK_IF(
        ubBlockSize <= 0,
        OP_LOGE(context, "invalid ubBlockSize: %ld", ubBlockSize),
        return ge::GRAPH_FAILED);

    // 6. Core 间切分
    //
    // 小张量只使用一个核，避免多核调度开销；
    // 大张量尽可能使用所有可用 AIV Core。
    int64_t targetCoreNum = 1;

    if (totalNum >= MIN_SPLIT_THRESHOLD) {
        targetCoreNum = coreNum;
    }

    if (totalNum == 0) {
        // 空张量仍至少启动一个核，Kernel 内不执行实际循环
        tiling->blockFactor = 1;
        targetCoreNum = 1;
    } else {
        // 每个核处理的元素数，并按 UB Block 对齐
        tiling->blockFactor = CeilAlign(
            CeilDiv(totalNum, targetCoreNum),
            ubBlockSize);

        // 对齐后可能不再需要 targetCoreNum 个核，
        // 因此重新计算实际使用核数。
        targetCoreNum = CeilDiv(
            totalNum,
            static_cast<int64_t>(tiling->blockFactor));
    }

    // 7. Core 内 UB 切分
    //
    // ReLU 有一个输入和一个输出。
    // Double Buffer：
    //   inputQueue  × 2
    //   outputQueue × 2
    // 共需要 4 份 UB Tensor。
    const int64_t ubElementCapacity =
        FloorDiv(static_cast<int64_t>(ubSize), TYPE_SIZE);

    tiling->ubFactor = FloorAlign(
        FloorDiv(ubElementCapacity, UB_BUFFER_NUM),
        ubBlockSize);

    OP_CHECK_IF(
        tiling->ubFactor <= 0,
        OP_LOGE(
            context,
            "invalid ubFactor, ubSize=%lu, ubBlockSize=%ld",
            ubSize,
            ubBlockSize),
        return ge::GRAPH_FAILED);

    context->SetBlockDim(static_cast<uint32_t>(targetCoreNum));

    // 8. 根据 dtype 选择 Kernel 分支
    uint64_t tilingKey = 0;

    if (dataType == ge::DT_FLOAT16 ||
        dataType == ge::DT_BF16) {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(tilingKey);

    OP_LOGD(
        context,
        "Relu tiling: totalNum=%ld, blockFactor=%ld, "
        "ubFactor=%ld, blockDim=%ld, ubSize=%lu",
        totalNum,
        static_cast<int64_t>(tiling->blockFactor),
        static_cast<int64_t>(tiling->ubFactor),
        targetCoreNum,
        ubSize);

    return ge::GRAPH_SUCCESS;

}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
