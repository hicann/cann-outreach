#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/add_tiling.h"
#include "../op_kernel/tiling_key_add.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    const gert::Tensor *tensorY = context->GetRequiredInputTensor(1);
    if (tensorX == nullptr || tensorY == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtype = tensorX->GetDataType();
    if ((dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) ||
        tensorY->GetDataType() != dtype) {
        return ge::GRAPH_FAILED;
    }

    // 本题输入为 (8, 2048)，共 16384 个元素。
    constexpr uint32_t tileLength = 256;
    const auto length = tensorX->GetShapeSize();
    if (length <= 0 || length > UINT32_MAX ||
        length != tensorY->GetShapeSize() ||
        length % tileLength != 0) {
        return ge::GRAPH_FAILED;
    }

    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores <= 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t tileCount =
        static_cast<uint32_t>(length) / tileLength;

    uint32_t blockDim = static_cast<uint32_t>(availableCores);
    if (blockDim > tileCount) {
        blockDim = tileCount;
    }

    // 保证每个核获得整数个 tile，不产生尾块。
    while (tileCount % blockDim != 0) {
        --blockDim;
    }

    uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    AddTilingData *tiling = context->GetTilingData<AddTilingData>();
    tiling->length = static_cast<uint32_t>(length);
    tiling->tileLength = tileLength;

    context->SetBlockDim(blockDim);
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    if (inputShape == nullptr || outputShape == nullptr) {
        return GRAPH_FAILED;
    }

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

class Add : public OpDef {
public:
    explicit Add(const char *name) : OpDef(name)
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

OP_ADD(Add);

}  // namespace ops