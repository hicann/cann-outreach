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

static inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (b <= 0) ? 0 : (a + b - 1) / b;
}

static inline int64_t CeilAlign(int64_t a, int64_t b)
{
    return (b <= 0) ? 0 : (a + b - 1) / b * b;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 1. 设置 tiling key（根据输入 dtype 选择 half 或 float）
    uint32_t schMode = RELU_TPL_SCH_MODE_1;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        schMode = RELU_TPL_SCH_MODE_0;
    }
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 2. 获取输入 shape 计算总元素数量
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* inShape = context->GetInputShape(0);
    int64_t totalNum = 1;
    if (inShape != nullptr) {
        for (size_t i = 0; i < inShape->GetStorageShape().GetDimNum(); ++i) {
            totalNum *= inShape->GetStorageShape().GetDim(i);
        }
    }

    // 3. 多核分块与 UB 切分
    constexpr int64_t TARGET_CORE_ELEMS = 2048;
    constexpr int64_t DEFAULT_UB_FACTOR = 2048;
    constexpr int64_t MAX_CORES = 8;

    int64_t usedCoreNum = 1;
    int64_t blockFactor = totalNum;
    int64_t ubFactor = DEFAULT_UB_FACTOR;

    if (totalNum <= TARGET_CORE_ELEMS) {
        usedCoreNum = 1;
        blockFactor = totalNum;
        ubFactor = CeilAlign(totalNum, 64);
        if (ubFactor < 64) {
            ubFactor = 64;
        }
    } else {
        int64_t neededCores = CeilDiv(totalNum, TARGET_CORE_ELEMS);
        if (neededCores <= MAX_CORES) {
            usedCoreNum = (neededCores > 0) ? neededCores : 1;
        } else {
            usedCoreNum = MAX_CORES;
        }
        blockFactor = CeilAlign(CeilDiv(totalNum, usedCoreNum), 64);
        ubFactor = DEFAULT_UB_FACTOR;
    }

    // 4. 设置 tiling 数据和 BlockDim
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCoreNum);

    // 5. 配置 workspace 大小（Relu 矢量算子无需额外系统 workspace）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
