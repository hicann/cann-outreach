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

constexpr uint32_t WS_SYS_SIZE = 0U;

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 获取输入 shape
    const gert::StorageShape* storage_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, storage_shape);
    const gert::Shape input_shape = storage_shape->GetShape();
    
    // 计算总元素数
    int64_t totalNum = 1;
    for (size_t i = 0; i < input_shape.GetDimNum(); i++) {
        totalNum *= input_shape.GetDim(i);
    }

    // 获取数据类型
    auto inputDesc = context->GetInputDesc(0);
    int64_t dataSize = 4;
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
        dataSize = 2;
    }

    // 获取平台信息
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    int64_t coreNum = 1;
    uint64_t ubSize = 0;
    if (platformInfoPtr != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (coreNum < 1) {
            coreNum = 1;
        }
    }
    
    if (ubSize == 0) {
        ubSize = 1024 * 1024;
    }

    // 关键修复：数据量较小时（<= 4096），使用单核
    // 避免多核分配不均导致错误
    int64_t usedCoreNum = (totalNum <= 4096) ? 1 : coreNum;

    // 每个核处理的数据量
    int64_t perCoreLength = totalNum / usedCoreNum;
    if (perCoreLength < 1) {
        perCoreLength = 1;
    }

    // 计算 tile 大小
    int64_t maxTileElements = static_cast<int64_t>((ubSize / dataSize) / 2);
    if (maxTileElements < 1) {
        maxTileElements = 1;
    }

    int64_t tileLength = (maxTileElements < perCoreLength) ? maxTileElements : perCoreLength;
    if (tileLength < 1) {
        tileLength = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = tileLength;
    tiling->ubFactor = 0;

    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = WS_SYS_SIZE;

    // 设置 BlockDim
    context->SetBlockDim(usedCoreNum);

    uint64_t tilingKey;
    if (inputDesc != nullptr && inputDesc->GetDataType() == ge::DT_FLOAT16) {
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