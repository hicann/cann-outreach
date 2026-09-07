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
    // ================================================================
    // 阶段1：获取平台信息（UB 大小、AIV 核数）
    // ================================================================
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    // ================================================================
    // 阶段2：获取输入 shape 与 dtype
    // ================================================================
    // 逐维相乘得到总元素数（标量 shape 的 dimNum 为 0，视为 1 个元素）
    const gert::StorageShape* inputShapeX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapeX);
    int64_t totalIdx = 1;
    for (int64_t i = 0; i < inputShapeX->GetStorageShape().GetDimNum(); i++) {
        totalIdx *= inputShapeX->GetStorageShape().GetDim(i);
    }
    OP_CHECK_IF(totalIdx <= 0, OP_LOGE(context, "totalIdx is invalid"), return ge::GRAPH_FAILED);

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    OP_CHECK_IF(
        dataType != ge::DT_FLOAT && dataType != ge::DT_FLOAT16,
        OP_LOGE(context, "invalid dtype"),
        return ge::GRAPH_FAILED);
    int64_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;

    // ================================================================
    // 阶段3：设置 Workspace 大小
    // ================================================================
    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // ================================================================
    // 阶段4：计算 tiling 数据并下发到 kernel 侧
    // ================================================================
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // totalNum：总元素数
    tiling->totalNum = totalIdx;
    // blockFactor：每核处理的元素数（向上取整均分到尽量多的核）
    tiling->blockFactor = Ops::Base::CeilDiv(totalIdx, coreNum);
    // 实际使用的核数（尾核只处理剩余部分，kernel 侧据此做边界处理）
    int64_t usedCoreNum = Ops::Base::CeilDiv(totalIdx, tiling->blockFactor);
    context->SetBlockDim(usedCoreNum);

    // ubFactor：UB 单次循环处理的元素数
    // 1 输入 + 1 输出 × 双缓冲 = 4 块 UB tensor；
    // 向下对齐到 32 字节（float 为 8 个元素，half 为 16 个），满足矢量指令对齐要求
    constexpr int64_t BUFFER_NUM_TOTAL = 4;
    int64_t alignElems = 32 / typeSize; //alignElems个数字是一个单元
    tiling->ubFactor = Ops::Base::FloorAlign(
        Ops::Base::FloorDiv(static_cast<int64_t>(ubSize), typeSize * BUFFER_NUM_TOTAL), alignElems); // 以32B为一个单元，UB的限制一次智能处理几个x元素
    OP_CHECK_IF(tiling->ubFactor <= 0, OP_LOGE(context, "ubFactor is invalid"), return ge::GRAPH_FAILED);

    // 根据输入 dtype 选择 tilingKey（与 op_kernel/relu.cpp 的模板分支对应：
    // MODE_0 -> half，MODE_1 -> float）
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
