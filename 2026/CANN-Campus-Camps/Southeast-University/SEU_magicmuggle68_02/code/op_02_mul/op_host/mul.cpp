// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    int32_t availableCoreNum =
        platform.GetCoreNumAiv();

    const gert::Tensor *tensorX =
        context->GetRequiredInputTensor(0);

    uint32_t length =
        tensorX->GetShapeSize();

    // 名称必须与tiling_key_mul.h中的DT_X一致
    uint32_t DT_X =
        static_cast<uint32_t>(
            tensorX->GetDataType());

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->length = length;

    uint32_t blockDim =
        availableCoreNum > 0
            ? static_cast<uint32_t>(availableCoreNum)
            : 1;

    constexpr uint32_t ALIGN_ELEMENTS = 16;

    while (blockDim > 1) {
        bool divisible =
            length % blockDim == 0;

        bool aligned =
            divisible &&
            (length / blockDim) % ALIGN_ELEMENTS == 0;

        if (aligned) {
            break;
        }

        --blockDim;
    }

    context->SetBlockDim(blockDim);

    size_t *workspace =
        context->GetWorkspaceSizes(1);

    workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape =
        context->GetInputShape(0);

    gert::Shape *zShape =
        context->GetOutputShape(0);

    *zShape = *xShape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    const auto inputDataType =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        inputDataType);

    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};


OP_ADD(Mul);

}  // namespace ops