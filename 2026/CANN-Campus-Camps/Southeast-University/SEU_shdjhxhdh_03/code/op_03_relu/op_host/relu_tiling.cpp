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

const uint32_t TILE_NUM = 16;
const uint32_t BLOCK_NUM = 8;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    /*
     * =========================================================
     * 1. 获取输入数据类型
     * =========================================================
     */

    auto inputDesc = context->GetInputDesc(0);

    ge::DataType dataType = inputDesc->GetDataType();

    /*
     * schMode 与 kernel 中的模板类型必须严格对应：
     *
     * schMode = 0 --> half
     * schMode = 1 --> float
     */

    uint32_t schMode = 0;

    if (dataType == ge::DT_FLOAT16) {
        schMode = RELU_TPL_SCH_MODE_0;
    } else if (dataType == ge::DT_FLOAT) {
        schMode = RELU_TPL_SCH_MODE_1;
    } else {
        return ge::GRAPH_FAILED;
    }

    /*
     * 选择模板参数
     */
    ASCENDC_TPL_SEL_PARAM(context, schMode);


    /*
     * =========================================================
     * 2. 获取输入 Shape
     * =========================================================
     */

    const gert::StorageShape* x_shape =
        context->GetInputShape(0);

    uint32_t data_sz = 1;

    for (int i = 0;
         i < x_shape->GetStorageShape().GetDimNum();
         i++) {

        data_sz *=
            x_shape->GetStorageShape().GetDim(i);
    }


    /*
     * =========================================================
     * 3. 写入 TilingData
     * =========================================================
     */

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    tiling->totalLength = data_sz;
    tiling->tileNum = TILE_NUM;


    /*
     * =========================================================
     * 4. 设置核数量
     * =========================================================
     */

    context->SetBlockDim(BLOCK_NUM);


    /*
     * =========================================================
     * 5. Workspace
     * =========================================================
     */

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