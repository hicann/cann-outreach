#include "../op_kernel/div_custom_template_tiling.h"
#include "register/op_def_registry.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace optiling {
namespace {
constexpr uint32_t MAX_BLOCK_DIM = 8;
constexpr uint32_t MAX_TILE_LENGTH = 1024;
constexpr uint32_t DTYPE_FLOAT16 = 0;
constexpr uint32_t DTYPE_FLOAT32 = 1;
constexpr uint32_t DATA_COPY_ALIGNMENT_BYTES = 32;

uint32_t SelectBlockDim(uint32_t totalLength, uint32_t alignmentElements)
{
    const uint32_t alignedUnits = totalLength / alignmentElements;
    uint32_t blockDim = std::min(MAX_BLOCK_DIM, alignedUnits);
    while (blockDim > 1 && alignedUnits % blockDim != 0) {
        --blockDim;
    }
    return blockDim;
}
}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *xShape = context->GetInputShape(0);
    const auto *yShape = context->GetInputShape(1);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto &xStorageShape = xShape->GetStorageShape();
    const auto &yStorageShape = yShape->GetStorageShape();
    if (xStorageShape.GetDimNum() != yStorageShape.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    for (int32_t index = 0; index < xStorageShape.GetDimNum(); ++index) {
        if (xStorageShape.GetDim(index) != yStorageShape.GetDim(index)) {
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t xLength = xStorageShape.GetShapeSize();
    const int64_t yLength = yStorageShape.GetShapeSize();
    if (xLength <= 0 || xLength != yLength ||
        static_cast<uint64_t>(xLength) > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    const auto *xDesc = context->GetInputDesc(0);
    const auto *yDesc = context->GetInputDesc(1);
    if (xDesc == nullptr || yDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto xDataType = xDesc->GetDataType();
    const auto yDataType = yDesc->GetDataType();
    if (xDataType != yDataType || (xDataType != ge::DT_FLOAT16 && xDataType != ge::DT_FLOAT)) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength = static_cast<uint32_t>(xLength);
    const uint32_t elementBytes = xDataType == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t);
    const uint32_t alignmentElements = DATA_COPY_ALIGNMENT_BYTES / elementBytes;
    if (totalLength % alignmentElements != 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t blockDim = SelectBlockDim(totalLength, alignmentElements);
    const uint32_t blockLength = totalLength / blockDim;
    uint32_t tileLength = std::min(MAX_TILE_LENGTH, blockLength);
    tileLength -= tileLength % alignmentElements;
    if (tileLength == 0) {
        tileLength = alignmentElements;
    }

    DivCustomTemplateTilingData tilingData;
    tilingData.set_totalLength(totalLength);
    tilingData.set_blockLength(blockLength);
    tilingData.set_tileLength(tileLength);
    tilingData.set_dataType(xDataType == ge::DT_FLOAT ? DTYPE_FLOAT32 : DTYPE_FLOAT16);

    context->SetBlockDim(blockDim);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class DivCustomTemplate : public OpDef {
public:
    explicit DivCustomTemplate(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(DivCustomTemplate);
}  // namespace ops
