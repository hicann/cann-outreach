/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */
#include "register/op_def_registry.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t CORE_ALIGN_BYTES = 512;  // 单位：字节
constexpr int64_t UB_ALIGN_BYTES = 256;
constexpr int64_t BUFFER_COUNT = 4;        // 输入、输出各两个 buffer

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (coreNum <= 0 || ubSize == 0) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;
    if (GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS ||
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto inputShape = context->GetInputShape(0);
    auto inputDesc = context->GetInputDesc(0);
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    if (inputShape == nullptr || inputDesc == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto dtype = inputDesc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const int64_t typeSize = dtype == ge::DT_FLOAT16 ? 2 : TYPE_SIZE;
    const int64_t totalNum = EnsureNotScalar(inputShape->GetStorageShape()).GetShapeSize();
    if (totalNum < 0) {
        return ge::GRAPH_FAILED;
    }

    // 约按每核 4 KiB 工作量选择核数，避免小输入启动过多核。
    const int64_t minCoreElements = MIN_SPLIT_THRESHOLD * TYPE_SIZE / typeSize;
    const int64_t coreAlign = CORE_ALIGN_BYTES / typeSize;
    int64_t usedCoreNum = 1;
    int64_t blockFactor = coreAlign;
    if (totalNum > 0) {
        usedCoreNum = CeilDiv(totalNum, minCoreElements);
        usedCoreNum = usedCoreNum < coreNum ? usedCoreNum : coreNum;
        blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), coreAlign);
        usedCoreNum = CeilDiv(totalNum, blockFactor);
    }

    // 双缓冲的四份空间一起计入 UB 预算；ubFactor 单位为元素。
    const int64_t ubAlign = UB_ALIGN_BYTES / typeSize;
    int64_t ubFactor = FloorAlign(static_cast<int64_t>(ubSize / typeSize / BUFFER_COUNT), ubAlign);
    if (ubFactor <= 0) {
        return ge::GRAPH_FAILED;
    }
    ubFactor = ubFactor < blockFactor ? ubFactor : blockFactor;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 保留题目给出的 FP16 / FP32 tiling key 对应关系。
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16) {
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
