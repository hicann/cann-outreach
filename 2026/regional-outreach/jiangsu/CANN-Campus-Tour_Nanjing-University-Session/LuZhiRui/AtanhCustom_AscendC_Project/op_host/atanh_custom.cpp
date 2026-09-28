#include "atanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AtanhCustomTilingData tiling;

    auto shape = context->GetInputShape(0)->GetStorageShape();
    uint32_t total =
        static_cast<uint32_t>(shape.GetShapeSize());

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) cores = 1;

    uint32_t perCore =
        (total + cores - 1) / cores;

    perCore = (perCore + 255) / 256 * 256;

    uint32_t blocks =
        (total + perCore - 1) / perCore;

    uint32_t tile = 8192;

    ge::Shape tmpShape({static_cast<int64_t>(tile)});
    uint32_t maxTmp = 0;
    uint32_t minTmp = 0;

    AscendC::GetAtanhMaxMinTmpSize(
        tmpShape,
        sizeof(half),
        false,
        maxTmp,
        minTmp);

    tiling.set_blockNum(blocks);
    tiling.set_totalLength(total);
    tiling.set_numPerCore(perCore);
    tiling.set_tileLength(tile);
    tiling.set_tmpBufferSize(minTmp);

    tiling.SaveToBuffer(
        context->GetRawTilingData()->GetData(),
        context->GetRawTilingData()->GetCapacity());

    context->GetRawTilingData()->SetDataSize(
        tiling.GetDataSize());

    context->SetBlockDim(blocks);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape(
    gert::InferShapeContext* context)
{
    *context->GetOutputShape(0) =
        *context->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferType(
    gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(
        0,
        context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}
