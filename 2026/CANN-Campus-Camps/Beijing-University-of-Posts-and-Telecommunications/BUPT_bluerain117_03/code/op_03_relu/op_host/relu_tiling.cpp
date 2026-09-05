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
constexpr int64_t HALF_TYPE_SIZE = 2;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024; // 每核最少处理的元素数，避免小数据过度切分
constexpr int64_t UB_BLOCK_BYTES = 32;        // UB 搬运 32B 对齐粒度
// UB 内单次搬运元素上限的推导：in/out 两个队列 × 每队列 BUFFER_NUM(2) 双缓冲 × 2 倍余量
constexpr int64_t UB_QUEUE_NUM = 2;
constexpr int64_t UB_BUFFER_NUM = 2;
constexpr int64_t UB_RESERVE_FACTOR = 2;

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

    // 输入 shape 逐维相乘得到总元素数（标量按 1 个元素处理）
    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape xShape = EnsureNotScalar(xStorageShape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); i++) {
        totalNum *= xShape.GetDim(i);
    }
    OP_CHECK_IF(totalNum < 0, OP_LOGE(context, "totalNum must be positive"), return ge::GRAPH_FAILED);

    // 按 dtype 确定单元素字节数与 32B 对齐粒度（元素个数）
    auto inputDesc = context->GetInputDesc(0);
    int64_t dtypeSize = TYPE_SIZE;
    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        dtypeSize = HALF_TYPE_SIZE;
    }
    const int64_t alignNum = UB_BLOCK_BYTES / dtypeSize;

    // 核间切分：每核至少处理 MIN_SPLIT_THRESHOLD 个元素；目标形状 (8,2048)=16384 时 16 核并行
    int64_t blockDim = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    blockDim = blockDim < coreNum ? blockDim : coreNum;
    blockDim = blockDim < 1 ? 1 : blockDim;

    // 每核处理量按 32B 向上对齐，保证各核 GM 起始地址对齐；尾部核剩余量由 kernel 兜底
    const int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignNum);

    // 核内切分：UB 单次搬运元素上限（保证 in/out 双队列 + 双缓冲仍落在 UB 内）
    int64_t ubFactorLimit = FloorAlign(
        static_cast<int64_t>(ubSize) / (dtypeSize * UB_QUEUE_NUM * UB_BUFFER_NUM * UB_RESERVE_FACTOR), alignNum);
    // 防御：避免 UB 异常小时 ubFactor 为 0 导致 kernel 侧除零
    ubFactorLimit = ubFactorLimit < alignNum ? alignNum : ubFactorLimit;
    const int64_t ubFactor = ubFactorLimit < blockFactor ? ubFactorLimit : blockFactor;

    // 设置 tiling 数据
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
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