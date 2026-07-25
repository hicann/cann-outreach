#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t CORE_NUM = 40U;
constexpr uint32_t TILE_NUM = 2U;
}

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    AddCustomTemplateTilingData *tilingData =
        context->GetTilingData<AddCustomTemplateTilingData>();

    const uint32_t totalLength = static_cast<uint32_t>(
        context->GetInputShape(0)
            ->GetOriginShape()
            .GetShapeSize());

    context->SetBlockDim(CORE_NUM);

    tilingData->totalLength = totalLength;
    tilingData->tileNum = TILE_NUM;

    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape =
        context->GetInputShape(0);

    gert::Shape *zShape =
        context->GetOutputShape(0);

    *zShape = *xShape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    const auto xDataType =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        xDataType);

    return GRAPH_SUCCESS;
}
}

namespace ops {
class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char *name)
        : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore()
            .AddConfig("ascend910b");
    }
};

OP_ADD(AddCustomTemplate);
}