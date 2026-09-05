#include "register/tilingdata_base.h"
#include "register/op_def_registry.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ReluTilingData)

    TILING_DATA_FIELD_DEF(uint32_t, totalElem);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
    TILING_DATA_FIELD_DEF(uint32_t, blockLen);
    TILING_DATA_FIELD_DEF(uint32_t, tileNumPerCore);
    TILING_DATA_FIELD_DEF(uint32_t, tileLen);

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Relu, ReluTilingData)

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    ReluTilingData tiling;

    const gert::StorageShape* xShape =
        context->GetInputShape(0);

    uint32_t totalElem =
        xShape->GetOriginShape().GetShapeSize();

    const uint32_t CORE_NUM = 8;

    uint32_t blockLen =
        totalElem / CORE_NUM;

    const uint32_t TILE_NUM_PER_CORE = 8;

    uint32_t tileLen =
        blockLen / TILE_NUM_PER_CORE;

    tiling.set_totalElem(totalElem);
    tiling.set_coreNum(CORE_NUM);
    tiling.set_blockLen(blockLen);
    tiling.set_tileNumPerCore(TILE_NUM_PER_CORE);
    tiling.set_tileLen(tileLen);

    context->SetBlockDim(CORE_NUM);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());

    context->GetRawTilingData()->SetDataSize(
        tiling.GetDataSize());

    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

IMPL_OP_OPTILING(Relu).Tiling(optiling::ReluTilingFunc);