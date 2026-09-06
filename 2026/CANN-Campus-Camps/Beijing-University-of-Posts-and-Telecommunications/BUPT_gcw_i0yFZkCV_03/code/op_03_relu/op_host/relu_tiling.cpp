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
// 与 op_kernel/relu.h 中 BUFFER_NUM 保持一致（双缓冲）
constexpr int64_t HOST_BUFFER_NUM = 2;
// UB 预算安全余量：实际可用量按 90% 计，防止 InitBuffer 触顶
constexpr int64_t UB_SAFE_RATIO_NUM = 9;
constexpr int64_t UB_SAFE_RATIO_DEN = 10;

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
    // 平台信息：AIV 核数与单核 UB 容量
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

    // 输入信息：dtype 决定元素字长与对齐元素数，shape 决定切分
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    // 本算子仅注册 DT_FLOAT / DT_FLOAT16（见 relu_def.cpp），字长直接按 dtype 判断
    int64_t typeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;
    OP_CHECK_IF(typeSize <= 0, OP_LOGE(context, "invalid dtype size"), return ge::GRAPH_FAILED);
    const gert::Tensor* xTensor = context->GetRequiredInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xTensor);
    int64_t totalNum = static_cast<int64_t>(xTensor->GetShapeSize());

    uint32_t ubBlockSize = GetUbBlockSize(context);        // 32B：DataCopy 最小对齐块（模板函数需传 context）
    int64_t alignElems = static_cast<int64_t>(ubBlockSize) / typeSize;  // fp16=16, fp32=8

    if (totalNum <= 0) {
        // 空输入防御：直接以最小配置返回
        tiling->totalNum = 0;
        tiling->blockFactor = alignElems;
        tiling->ubFactor = alignElems;
        context->SetBlockDim(1);
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
        return ge::GRAPH_SUCCESS;
    }

    // ---- 核间切分 ----
    // blockFactor = ceil(totalNum / coreNum) 向上对齐到 32B 元素数：
    //   保证每核 GM 窗口起始地址 (blockIdx * blockFactor) 始终 32B 对齐，
    //   且尾核的剩余量 (totalNum - start) 也是 alignElems 的倍数，CopyIn/CopyOut 无需 mask。
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignElems);

    // ---- 核内切分 ----
    // ubFactor 受 UB 预算约束：2 个队列(x/y) × BUFFER_NUM 份双缓冲 × 元素字长 ≤ ubSize，
    // 另留 10% 安全余量，防止 InitBuffer 触顶。
    int64_t ubFactor = static_cast<int64_t>(ubSize) / (2 * HOST_BUFFER_NUM * typeSize);
    ubFactor = ubFactor / UB_SAFE_RATIO_NUM * UB_SAFE_RATIO_DEN;
    ubFactor = FloorAlign(ubFactor, alignElems);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;   // 单核数据一轮装得下就不必多轮
    }
    if (ubFactor < alignElems) {
        ubFactor = alignElems;    // 极小输入的兜底下限
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 启动核数：恰好覆盖数据所需的核数，避免空转核
    int64_t usedCores = CeilDiv(totalNum, blockFactor);
    context->SetBlockDim(static_cast<uint32_t>(usedCores));

    // 根据输入 dtype 选择 tilingKey
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
