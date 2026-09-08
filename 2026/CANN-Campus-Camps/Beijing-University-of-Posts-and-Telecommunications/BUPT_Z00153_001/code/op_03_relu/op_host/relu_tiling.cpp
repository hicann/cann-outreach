/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"

namespace optiling {

// tileNum 语义 = 每核总轮数（kernel 侧 tileLength = blockLength / tileNum）
static const uint32_t TILE_NUM_FP32 = 2; // fp32: 2 轮 x 1024 元素双缓冲
static const uint32_t TILE_NUM_FP16 = 1; // fp16: 1 轮 x 2048 元素单块大 burst

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 输入 shape 逐维相乘得到总元素数
    const gert::StorageShape* x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t data_sz = 1;
    for (int i = 0; i < x_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = static_cast<uint32_t>(data_sz);

    // dtype 编码写入 tiling：0=fp32, 1=fp16（kernel 侧据此选择实现分支）
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    bool isFp32 = (inputDesc->GetDataType() == ge::DT_FLOAT);
    tiling->dtype = isFp32 ? 0 : 1;
    // 轮数按 dtype 配置
    tiling->tileNum = isFp32 ? TILE_NUM_FP32 : TILE_NUM_FP16;

    // 配置启动核数：固定 8（本环境 blockDim 超过物理核数会被切批串行，反而变慢）
    context->SetBlockDim(8);

    // 配置 workspace 大小
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
