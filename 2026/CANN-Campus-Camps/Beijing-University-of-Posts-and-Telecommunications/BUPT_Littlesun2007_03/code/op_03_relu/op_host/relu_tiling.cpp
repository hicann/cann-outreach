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
    // TODO: 实现 Tiling 逻辑
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

    // TODO: 设置 tiling 数据
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    
    // 1. 获取总数据量
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    
    int64_t data_sz = 1;
    auto shape_obj = x_shape->GetStorageShape();
    for (int i = 0; i < shape_obj.GetDimNum(); i++) {
        data_sz *= shape_obj.GetDim(i);
    }
    tiling->totalNum = data_sz;

    // 2. 获取数据类型大小
    int64_t typeSize = 2; // 默认 float16/bf16
    if (inputDesc->GetDataType() == ge::DT_FLOAT) {
        typeSize = 4;
    }

    // 3. 计算 ubFactor (单次搬运长度)
    double bufferRatio = 0.5; 
    int64_t maxElementsPerBuffer = static_cast<int64_t>((ubSize * bufferRatio) / 2 / typeSize);
    tiling->ubFactor = (maxElementsPerBuffer / 32) * 32;
    if (tiling->ubFactor == 0) tiling->ubFactor = 32;

    // 4. 【核心逻辑】动态计算 BlockDim 和 blockFactor
    
    // 第一步：估算每个核至少处理多少数据（这里简单设为 ubFactor，保证每个核至少跑一次完整的 UB 循环）
    int64_t minDataPerCore = tiling->ubFactor;
    
    // 第二步：计算理论需要的核数
    int64_t neededCores = (data_sz + minDataPerCore - 1) / minDataPerCore;
    
    // 第三步：限制核数不超过硬件最大核数，且至少为 1
    if (neededCores > coreNum) neededCores = coreNum;
    if (neededCores < 1) neededCores = 1;

    // 第四步：根据实际核数，重新计算每个核处理的数据量 (blockFactor)
    // 向上取整，确保所有数据都能被分配出去
    int64_t blockFactor = (data_sz + neededCores - 1) / neededCores;
    
    // 第五步：对齐到 32 的倍数（Ascend 矢量指令要求）
    blockFactor = (blockFactor / 32) * 32;
    if (blockFactor == 0) blockFactor = 32;

    // 赋值给 Tiling 结构体
    tiling->blockFactor = blockFactor;
    
    // 设置实际的核数
    context->SetBlockDim(neededCores);

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
