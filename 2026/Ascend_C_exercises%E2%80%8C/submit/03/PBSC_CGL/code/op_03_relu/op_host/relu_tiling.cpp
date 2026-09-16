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

constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t BLOCK_LENGTH = 2048;
constexpr uint32_t TILE_NUM = 1;

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    /*
     * 题目固定：
     *
     * shape = (8, 2048)
     * total = 16384
     *
     * 8 Core:
     * 每Core = 2048
     *
     * 不再核内切小Tile。
     */
    context->SetBlockDim(BLOCK_DIM);

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    tiling->blockLength = BLOCK_LENGTH;
    tiling->tileNum = TILE_NUM;

    /*
     * 获取x的数据类型：
     *
     * DT_FLOAT
     * DT_FLOAT16
     *
     * 并通过官方模板参数机制生成TilingKey。
     */
    ge::DataType dtypeX =
        context->GetInputDesc(0)->GetDataType();

    uint32_t DT_X =
        static_cast<uint32_t>(dtypeX);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_X);

    /*
     * Relu基础Vector API不需要额外workspace。
     */
    size_t* workspaceSize =
        context->GetWorkspaceSizes(1);

    workspaceSize[0] = 0;

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