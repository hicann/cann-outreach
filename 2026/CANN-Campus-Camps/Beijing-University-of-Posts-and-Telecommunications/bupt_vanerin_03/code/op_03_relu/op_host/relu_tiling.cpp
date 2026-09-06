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
constexpr int64_t DOUBLE_BUFFER_NUM = 2; // 与 op_kernel/relu.h 中 BUFFER_NUM 保持一致（队列双缓冲份数）

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

    // 输入 dtype -> 元素字节数（fp32=4, fp16/bf16=2）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType inputDtype = inputDesc->GetDataType();
    const bool isHalf = (inputDtype == ge::DT_FLOAT16 || inputDtype == ge::DT_BF16);
    const int64_t typeSize = isHalf ? static_cast<int64_t>(2) : TYPE_SIZE;

    // 展平总元素数（标量输入按 1 个元素处理）
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    // 空张量保护：按 1 个元素完成切分计算（避免除零），kernel 侧拿到 totalNum=0 会直接返回
    const int64_t calcNum = totalNum > 0 ? totalNum : 1;

    // ---- 多核切分：每核至少处理 4KB（MIN_SPLIT_THRESHOLD * TYPE_SIZE 字节），核数不超过物理上限 ----
    int64_t usedCoreNum = 8;
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    // 每核元素数按 256B 向上对齐：保证各核 GM 起始地址与搬运长度 32B 对齐，走 DataCopy 快路径
    const int64_t alignElems = 256 / typeSize; // 64 (fp32) / 128 (fp16)
    const int64_t blockFactor = CeilAlign(CeilDiv(calcNum, usedCoreNum), alignElems);
    // 对齐放大后实际所需核数可能少于 usedCoreNum，按除法结果启动，避免空转核
    const int64_t blockDim = CeilDiv(calcNum, blockFactor);

    // ---- UB 切分：输入/输出两条队列 + 双缓冲，2 * DOUBLE_BUFFER_NUM * ubFactor * typeSize <= ubSize ----
    int64_t ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / (2 * DOUBLE_BUFFER_NUM * typeSize), alignElems);
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor; // 单核数据可一次装入 UB 时 tile 取整块，核内循环仅 1 次
    }
    if (ubFactor < alignElems) {
        ubFactor = alignElems;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey：半精度走 schMode=0（half kernel），其余走 schMode=1（float kernel）
    const uint64_t tilingKey = isHalf ? GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
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
