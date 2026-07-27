#include <cstdint>
#include <limits>

#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t BLOCK_DIM = 40;
constexpr uint32_t TILE_NUM = 2;
constexpr uint32_t BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    AddCustomTemplateTilingData* tiling = context->GetTilingData<AddCustomTemplateTilingData>();
    if (inputShape == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t totalLength = inputShape->GetOriginShape().GetShapeSize();
    constexpr uint64_t elementsPerGroup =
        static_cast<uint64_t>(BLOCK_DIM) * TILE_NUM * BUFFER_NUM;
    if (totalLength <= 0 ||
        static_cast<uint64_t>(totalLength) > std::numeric_limits<uint32_t>::max() ||
        static_cast<uint64_t>(totalLength) % elementsPerGroup != 0) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(BLOCK_DIM);
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    // For [45, 20480], each of the 40 vector cores handles 23040 elements.
    // Two logical tiles plus double buffering produce four 5760-element
    // transfers per core (23040 bytes for float), reducing loop overhead while
    // keeping every transfer large enough to use GM bandwidth efficiently.
    tiling->tileNum = TILE_NUM;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char* name) : OpDef(name)
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
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AddCustomTemplate);
}
