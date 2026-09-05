/*!
 * \file relu_tiling.cpp
 * \brief ReLU 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint64_t WS_SYS_SIZE = 0;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto platform = platform_ascendc::PlatformAscendC(platformInfo);
    int64_t coreNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(coreNum <= 0 || ubSize == 0,
        OP_LOGE(context, "invalid platform resource"),
        return ge::GRAPH_FAILED);

    // 将输入 shape 的各维相乘，得到逐元素处理的总长度。
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t totalNum = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); ++i) {
        totalNum *= xShape->GetStorageShape().GetDim(i);
    }

    // 限制每个核至少处理 MIN_SPLIT_THRESHOLD 个元素，避免产生过小且未对齐的搬运块。
    int64_t splitCoreNum = FloorDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (splitCoreNum < 1) {
        splitCoreNum = 1;
    }
    if (coreNum > splitCoreNum) {
        coreNum = splitCoreNum;
    }
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    auto inputDesc = context->GetInputDesc(0);
    int64_t typeSize = 4;
    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
         inputDesc->GetDataType() == ge::DT_BF16)) {
        typeSize = 2;
    }
    int64_t elementsPerUb = ubSize / (BUFFER_NUM * 2 * typeSize);
    int64_t alignElements = GetUbBlockSize(context) / typeSize;
    int64_t ubFactor = FloorAlign(elementsPerUb, alignElements);
    if (ubFactor <= 0) {
        ubFactor = alignElements;
    }

    tiling->totalNum = static_cast<uint32_t>(totalNum);
    tiling->blockFactor = static_cast<uint32_t>(blockFactor);
    tiling->ubFactor = static_cast<uint32_t>(ubFactor);
    context->SetBlockDim(coreNum);

    // ReLU 不使用额外 workspace。
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;

    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
         inputDesc->GetDataType() == ge::DT_BF16)) {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0));
    } else {
        context->SetTilingKey(GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingParse(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(ReluTilingParse);

} // namespace optiling
