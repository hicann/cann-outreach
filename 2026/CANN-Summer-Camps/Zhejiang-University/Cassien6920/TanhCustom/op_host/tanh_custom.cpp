#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"

#include <cstring>
#include <limits>

namespace optiling {
constexpr uint32_t BLOCK_DIM = 8;
constexpr uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* xShape = context->GetInputShape(0);
    if (xShape == nullptr || context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t totalLength = xShape->GetStorageShape().GetShapeSize();
    // Eight cores, eight tiles and double buffering require a multiple of 128
    // elements. For FP16 this also keeps each tile 32-byte aligned.
    constexpr uint32_t ELEMENTS_PER_CYCLE = BLOCK_DIM * TILE_NUM * 2;
    if (totalLength <= 0 ||
        totalLength > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
        totalLength % ELEMENTS_PER_CYCLE != 0) {
        return ge::GRAPH_FAILED;
    }

    TanhCustomTilingData tilingData = {
        static_cast<uint32_t>(totalLength),
        TILE_NUM
    };
    auto* rawTilingData = context->GetRawTilingData();
    if (rawTilingData->GetCapacity() < sizeof(TanhCustomTilingData)) {
        return ge::GRAPH_FAILED;
    }
    std::memcpy(rawTilingData->GetData(), &tilingData, sizeof(TanhCustomTilingData));
    rawTilingData->SetDataSize(sizeof(TanhCustomTilingData));

    context->SetBlockDim(BLOCK_DIM);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class TanhCustom : public OpDef {
public:
    explicit TanhCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(TanhCustom);
}  // namespace ops
