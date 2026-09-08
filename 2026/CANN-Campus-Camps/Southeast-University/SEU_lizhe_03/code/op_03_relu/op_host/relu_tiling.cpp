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

const uint32_t TILE_NUM = 8;

// static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
// {
//     // 1. 获取输入数据类型
//     auto inputDesc = context->GetInputDesc(0);
//     uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());

//     // float / float16 对应不同模板实例
//     ASCENDC_TPL_SEL_PARAM(context, DT_X);

//     // 2. 获取 TilingData
//     ReluTilingData* tiling =
//         context->GetTilingData<ReluTilingData>();

//     // 3. 根据输入 shape 计算总元素数量
//     const gert::StorageShape* xShape =
//         context->GetInputShape(0);

//     uint32_t totalLength = 1;

//     for (int i = 0;
//          i < xShape->GetStorageShape().GetDimNum();
//          i++) {
//         totalLength *=
//             xShape->GetStorageShape().GetDim(i);
//     }

//     // 4. 写入 tiling 参数
//     tiling->totalLength = totalLength;
//     tiling->tileNum = TILE_NUM;

//     // 5. 使用 8 个 AI Core
//     context->SetBlockDim(8);

//     // 6. 不需要额外 workspace
//     size_t* currentWorkspace =
//         context->GetWorkspaceSizes(1);

//     currentWorkspace[0] = 0;

//     return ge::GRAPH_SUCCESS;
// }
static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 配置 Tiling 模板参数
    uint32_t schMode = RELU_TPL_SCH_MODE_0;
    ASCENDC_TPL_SEL_PARAM(context, schMode);

    // 获取 TilingData
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    // 根据输入 Shape 计算总元素数
    const gert::StorageShape* xShape =
        context->GetInputShape(0);

    uint32_t totalLength = 1;

    for (int i = 0;
         i < xShape->GetStorageShape().GetDimNum();
         i++) {
        totalLength *=
            xShape->GetStorageShape().GetDim(i);
    }

    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;

    // 8 核
    context->SetBlockDim(8);

    // 无额外 workspace
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling