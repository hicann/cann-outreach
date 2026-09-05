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
constexpr int64_t ALIGN_SIZE = 32;      // 元素对齐数
constexpr int64_t DEFAULT_UB_FACTOR = 2048;

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

    // 1. 计算总元素数
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    int64_t totalNum = 1;
    for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= x_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalNum = totalNum;

    // 2. 计算每个核分得的元素数（向上取整）
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    // 将 blockFactor 向上对齐到 ALIGN_SIZE 的整数倍，确保每个核处理的元素数是32的倍数
    blockFactor = CeilAlign(blockFactor, ALIGN_SIZE);
    tiling->blockFactor = blockFactor;

    // 3. 确定 ubFactor：先取默认值，并调整为能整除 blockFactor 且是 ALIGN_SIZE 的倍数
    int64_t ubFactor = DEFAULT_UB_FACTOR;
    // 如果 blockFactor 小于默认值，则直接取 blockFactor（已经是ALIGN_SIZE的倍数）
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }
    // 调整 ubFactor 使其能整除 blockFactor（或使 blockFactor 是 ubFactor 的整数倍）
    // 同时保证 ubFactor 是 ALIGN_SIZE 的倍数
    while (ubFactor > ALIGN_SIZE) {
        if (blockFactor % ubFactor == 0 && ubFactor % ALIGN_SIZE == 0) {
            break;
        }
        ubFactor /= 2;
    }
    // 确保 ubFactor 至少为 ALIGN_SIZE 的倍数
    if (ubFactor < ALIGN_SIZE) {
        ubFactor = ALIGN_SIZE;
    }
    // 如果 ubFactor 无法整除 blockFactor，则向上调整 blockFactor 使其为 ubFactor 的整数倍
    if (blockFactor % ubFactor != 0) {
        blockFactor = CeilDiv(blockFactor, ubFactor) * ubFactor;
        tiling->blockFactor = blockFactor;
    }
    tiling->ubFactor = ubFactor;

    // 4. 配置使用的核数
    context->SetBlockDim(coreNum);

    // 5. 根据输入 dtype 选择 tilingKey
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