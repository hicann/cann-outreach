/*!
 * \file square_tiling.cpp
 * \brief Square tiling implementation
 */

#include "register/op_impl_registry.h"

#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

static ge::graphStatus SquareTilingFunc(
    gert::TilingContext* context)
{
    // =====================================================
    // 1. 获取输入Tensor总元素数量
    // =====================================================

    const gert::StorageShape* inputShape =
        context->GetInputShape(0);

    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t totalNum =
        inputShape
            ->GetOriginShape()
            .GetShapeSize();

    if (totalNum <= 0) {
        return ge::GRAPH_FAILED;
    }


    // =====================================================
    // 2. 获取输入数据类型
    //
    // schMode:
    // 0 -> float16
    // 1 -> float32
    // =====================================================

    auto inputDesc =
        context->GetInputDesc(0);

    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtype =
        inputDesc->GetDataType();

    uint32_t schMode = 0;

    if (dtype == ge::DT_FLOAT16) {

        schMode = 0;

    } else if (dtype == ge::DT_FLOAT) {

        schMode = 1;

    } else {

        return ge::GRAPH_FAILED;
    }


    // =====================================================
    // 3. 多核切分
    //
    // 最多使用8个Vector Core。
    // totalNum可能不能被8整除，因此使用ceil除法。
    // =====================================================

    uint32_t targetCoreNum = 8;

    if (totalNum < 8) {
        targetCoreNum =
            static_cast<uint32_t>(totalNum);
    }


    int64_t blockFactor =
        (totalNum +
         static_cast<int64_t>(targetCoreNum) -
         1) /
        static_cast<int64_t>(targetCoreNum);


    // 再根据blockFactor算真正需要启动多少个核，
    // 保证不会启动完全没有数据的核。
    uint32_t usedCoreNum =
        static_cast<uint32_t>(
            (totalNum +
             blockFactor -
             1) /
            blockFactor);


    // =====================================================
    // 4. UB切分
    //
    // 每次最多处理1024个元素。
    //
    // float32:
    // 1024 * 4 = 4096 Byte
    //
    // float16:
    // 1024 * 2 = 2048 Byte
    //
    // 输入输出各一个Buffer，空间非常充足。
    // =====================================================

    constexpr int64_t UB_FACTOR = 1024;


    // =====================================================
    // 5. 填写TilingData
    // =====================================================

    SquareTilingData* tiling =
        context->GetTilingData<SquareTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        UB_FACTOR;


    // =====================================================
    // 6. 配置启动核数
    // =====================================================

    context->SetBlockDim(
        usedCoreNum);


    // =====================================================
    // 7. 配置TilingKey
    //
    // square_tiling_key.h不用改。
    // 根据schMode选择half / float Kernel。
    // =====================================================

    ASCENDC_TPL_SEL_PARAM(
        context,
        schMode);


    // =====================================================
    // 8. Square不需要额外Workspace
    // =====================================================

    size_t* workspace =
        context->GetWorkspaceSizes(1);

    if (workspace != nullptr) {
        workspace[0] = 0;
    }


    return ge::GRAPH_SUCCESS;
}


struct SquareCompileInfo {
};


static ge::graphStatus TilingParseForSquare(
    gert::TilingParseContext* context)
{
    (void)context;

    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc)
    .TilingParse<SquareCompileInfo>(
        TilingParseForSquare);


} // namespace optiling