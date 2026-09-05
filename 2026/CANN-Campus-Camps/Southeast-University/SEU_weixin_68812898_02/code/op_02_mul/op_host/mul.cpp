#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t length_x = tensor_x->GetShapeSize();

    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->length = length_x;

    context->SetBlockDim(num_cores_aiv);

    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape =
        context->GetInputShape(0);
    gert::Shape *outputShape =
        context->GetOutputShape(0);

    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    ge::DataType inputDataType =
        context->GetInputDataType(0);

    context->SetOutputDataType(0, inputDataType);

    return ge::GRAPH_SUCCESS;
}

}

namespace ops {

class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}