/*!
 * \file add_tiling.cpp
 * \brief Add 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/add_tiling_data.h"
#include "../op_kernel/add_tiling_key.h"

namespace optiling {
    
const uint32_t TILE_NUM = 16;

static ge::graphStatus AddTilingFunc(gert::TilingContext* context)
{
    // 获取输入 dtype，按模板参数选择 tilingKey（float -> float 实例，float16 -> float16 实例）
    auto inputDesc = context->GetInputDesc(0);
    uint32_t DT_X = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 输入 shape 逐维相乘得到总元素数
    AddTilingData* tiling = context->GetTilingData<AddTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    int32_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    tiling->totalLength = data_sz;
    tiling->tileNum = TILE_NUM;

    // 配置启动核数
    context->SetBlockDim(8);

    // 配置 workspace 大小
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForAdd([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct AddCompileInfo {};

IMPL_OP_OPTILING(Add).Tiling(AddTilingFunc).TilingParse<AddCompileInfo>(TilingParseForAdd);

} // namespace optiling
