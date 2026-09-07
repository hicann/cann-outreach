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
constexpr int64_t TYPE_SIZE = 4;          // 占位，实际根据 dtype 计算
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

    // 【修改点】：通过 GetInputShape 获取张量形状
    auto inShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inShapePtr);
    gert::Shape shape = EnsureNotScalar(inShapePtr->GetStorageShape());
    int64_t dimNum = shape.GetDimNum();
    int64_t totalNum = 1;
    for (int64_t i = 0; i < dimNum; ++i) {
        totalNum *= shape.GetDim(i);
    }

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = 0;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        typeSize = 2;
    } else if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
    } else {
        OP_LOGE(context, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }

    // 对齐元素数（32 字节对齐）
    const int64_t ALIGN_NUM = 32 / typeSize;
    int64_t totalAlignNum = CeilAlign(totalNum, ALIGN_NUM);

    // 多核分配
    int64_t usedCoreNum = std::min(coreNum, CeilDiv(totalAlignNum, ALIGN_NUM));
    if (usedCoreNum <= 0) {
        usedCoreNum = 1;
    }
    int64_t blocksPerCore = CeilDiv(totalAlignNum / ALIGN_NUM, usedCoreNum);
    int64_t alignBlocksPerCore = blocksPerCore * ALIGN_NUM;

    // UB 分块（双缓冲）
    uint64_t usableUbSize = ubSize / 2;
    int64_t maxUbElements = usableUbSize / typeSize;
    maxUbElements = FloorAlign(maxUbElements, ALIGN_NUM);
    if (maxUbElements <= 0) {
        maxUbElements = ALIGN_NUM;
    }
    if (maxUbElements > 4096) {
        maxUbElements = 4096;
    }

    // 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = alignBlocksPerCore;
    tiling->ubFactor = maxUbElements;

    context->SetBlockDim(usedCoreNum);

    // 根据 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
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