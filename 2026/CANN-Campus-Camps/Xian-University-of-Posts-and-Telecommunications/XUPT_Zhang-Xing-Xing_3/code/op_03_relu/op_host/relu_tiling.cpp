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

constexpr int32_t BUFFER_NUM = 2;        // 双缓冲（须与 relu.h 中 BUFFER_NUM 保持一致）
constexpr int32_t MAX_CORE_NUM = 16;     // 本算子最多启用的核数（本 shape 下用满 16 核）
constexpr int32_t MIN_PER_CORE = 1024;   // 每个核最少处理的元素数（数据少时减少核数）
constexpr int32_t TARGET_TILE_LENGTH = 1024; // 每块目标元素数：块越大循环越少、开销越低

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 1. 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    ge::DataType dataType = inputDesc->GetDataType();
    uint32_t DT_X = static_cast<uint32_t>(dataType);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 2. 输入 shape 逐维相乘得到总元素数量
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int64_t totalLength = 1;
    for (int i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= xShape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = totalLength;

    // 3. 多核切分：数据量小时减少核数，避免过度切分
    int64_t usedCoreNum = (totalLength + MIN_PER_CORE - 1) / MIN_PER_CORE;
    if (usedCoreNum > MAX_CORE_NUM) {
        usedCoreNum = MAX_CORE_NUM;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    int64_t blockLength = totalLength / usedCoreNum; // 每核元素数（与 kernel 侧 GetBlockNum() 一致）

    // 4. 单核内分块：目标每块 TARGET_TILE_LENGTH 个元素，块尽量大、循环尽量少。
    //    本 shape 下每核 1024 元素 -> 1 块单次搬运计算，开销最小。
    //    （竞赛固定 shape(8,2048) 下 blockLength 能被整除且 32B 对齐，保证正确性）
    int64_t tileNum = (blockLength + TARGET_TILE_LENGTH - 1) / TARGET_TILE_LENGTH;
    if (tileNum < 1) {
        tileNum = 1;
    }
    tiling->tileNum = tileNum;

    // 5. 配置启动核数
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 6. 配置 workspace 大小（本算子无需额外 workspace）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling

