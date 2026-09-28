#include <cstdint>
#include <cstddef>
#include <limits>

#include "add_custom_tiling.h"
#include "register/op_def_registry.h"

namespace {

constexpr uint32_t kTileLength = 128;

bool SameShape(const gert::Shape& lhs, const gert::Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

}  // namespace

namespace optiling {

ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* yShape = context->GetInputShape(1);
    if (xShape == nullptr || yShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& x = xShape->GetStorageShape();
    const gert::Shape& y = yShape->GetStorageShape();
    if (x.GetDimNum() != 2 || !SameShape(x, y)) {
        return ge::GRAPH_FAILED;
    }

    int64_t totalLength = 1;
    for (size_t i = 0; i < x.GetDimNum(); ++i) {
        const int64_t dim = x.GetDim(i);
        if (dim <= 0 || totalLength > std::numeric_limits<int64_t>::max() / dim) {
            return ge::GRAPH_FAILED;
        }
        totalLength *= dim;
    }
    if (totalLength > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    AddCustomTilingData tiling;
    tiling.set_totalLength(static_cast<uint32_t>(totalLength));
    tiling.set_tileLength(kTileLength);

    context->SetBlockDim(1);
    auto* rawTilingData = context->GetRawTilingData();
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    const gert::Shape* yShape = context->GetInputShape(1);
    gert::Shape* zShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr || zShape == nullptr ||
        xShape->GetDimNum() != 2 || !SameShape(*xShape, *yShape)) {
        return GRAPH_FAILED;
    }
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b").AddConfig("ascend310b");
    }
};

OP_ADD(AddCustom);

}  // namespace ops
