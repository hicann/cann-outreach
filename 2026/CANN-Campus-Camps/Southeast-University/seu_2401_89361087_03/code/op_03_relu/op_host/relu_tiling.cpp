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

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    // 获取输入shape
    const gert::StorageShape* inputShape =
        context->GetInputShape(0);

    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 输入总元素数量
    uint32_t totalLength =
        static_cast<uint32_t>(
            inputShape
                ->GetOriginShape()
                .GetShapeSize());

    // 获取输入数据类型
    const gert::CompileTimeTensorDesc* inputDesc =
        context->GetInputDesc(0);

    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtypeX =
        inputDesc->GetDataType();

    uint32_t DT_X =
        static_cast<uint32_t>(
            dtypeX);

    // 本题shape为(8, 2048)
    // 使用8个AI Vector Core
    context->SetBlockDim(8);

    // 获取TilingData
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 8 * 2048 = 16384
    tiling->totalLength =
        totalLength;

    // 每个核再切8块
    tiling->tileNum = 8;

    // 根据 float32 / float16 设置TilingKey
    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_X);

    // Relu不需要额外workspace
    size_t* workspace =
        context->GetWorkspaceSizes(1);

    if (workspace != nullptr) {
        workspace[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);

} // namespace optiling