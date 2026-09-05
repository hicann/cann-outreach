#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t numCores = platform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    int32_t dtypeSize = ge::GetSizeByDataType(dtypeX);
    uint32_t length = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    uint32_t blockDim = length < static_cast<uint32_t>(numCores) ? length : static_cast<uint32_t>(numCores);
    if (blockDim == 0) blockDim = 1;
    context->SetBlockDim(blockDim);

    constexpr uint32_t BUFFER_NUM = 2;
    uint64_t usableUb = ubSize > 8192 ? ubSize - 8192 : ubSize;
    uint32_t alignNum = 32 / static_cast<uint32_t>(dtypeSize);
    uint64_t tileLength = usableUb / (3ULL * BUFFER_NUM * static_cast<uint32_t>(dtypeSize));
    tileLength = tileLength / alignNum * alignNum;
    if (tileLength < alignNum) tileLength = alignNum;

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    tiling->length = length;
    tiling->tileLength = static_cast<uint32_t>(tileLength);

    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    auto inputType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputType);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name) {
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
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}