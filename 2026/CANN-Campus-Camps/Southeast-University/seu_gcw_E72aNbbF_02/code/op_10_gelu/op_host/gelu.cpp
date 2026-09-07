#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const gert::Tensor *input = context->GetRequiredInputTensor(0);
    const uint32_t length = static_cast<uint32_t>(input->GetShapeSize());
    const uint32_t platformCores = static_cast<uint32_t>(platform.GetCoreNumAiv());
    const uint32_t usefulCores = (length + 1023) / 1024;
    uint32_t coreNum = usefulCores < platformCores ? usefulCores : platformCores;
    coreNum = coreNum == 0 ? 1 : coreNum;

    const ge::DataType dtype = input->GetDataType();
    ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtype));

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->length = length;
    tiling->blockLength = (length + coreNum - 1) / coreNum;
    context->SetBlockDim(coreNum);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    *context->GetOutputShape(0) = *inputShape;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name) {
        Input("input_x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        Output("output").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}
