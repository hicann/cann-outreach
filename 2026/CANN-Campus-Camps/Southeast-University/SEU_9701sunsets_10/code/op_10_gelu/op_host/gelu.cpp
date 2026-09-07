#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNum = platform.GetCoreNumAiv();

    const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_input_x = tensor_input_x->GetDataType();

    uint32_t dtypeSize = ge::GetSizeByDataType(dtype_input_x);
    uint32_t length = tensor_input_x->GetShapeSize();
    uint32_t alignNum = 32 / dtypeSize;

    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    uint32_t usedCoreNum = 1;
    if (length > 8192) {
        usedCoreNum = (length + 4095) / 4096;
        usedCoreNum = usedCoreNum > static_cast<uint32_t>(coreNum)
                          ? static_cast<uint32_t>(coreNum)
                          : usedCoreNum;
    }

    uint32_t blockLength = (length + usedCoreNum - 1) / usedCoreNum;
    blockLength = (blockLength + alignNum - 1) / alignNum * alignNum;

    uint32_t tileLength;
    if (blockLength <= 2048) {
        tileLength = blockLength;
    } else if (blockLength <= 8192) {
        tileLength = 2048;
    } else {
        tileLength = 4096;
    }
    tileLength = (tileLength + alignNum - 1) / alignNum * alignNum;

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->length = length;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;

    // float32 用 erf，float16 用多项式
    tiling->mode = (length >= 32768 || dtype_input_x == ge::DT_FLOAT16) ? 1 : 0;

    context->SetBlockDim(usedCoreNum);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

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
    ge::DataType dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);
}  // namespace ops