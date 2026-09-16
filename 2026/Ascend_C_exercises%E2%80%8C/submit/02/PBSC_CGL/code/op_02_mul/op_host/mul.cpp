// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

constexpr uint32_t BLOCK_DIM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    uint32_t DT_X =
        static_cast<uint32_t>(tensor_x->GetDataType());

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    const uint32_t totalLength =
        static_cast<uint32_t>(tensor_x->GetShapeSize());

    // 固定8核
    context->SetBlockDim(BLOCK_DIM);

    // 每核直接处理一整行：2048 elements
    tiling->blockLength =
        totalLength / BLOCK_DIM;

    // 无workspace
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
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
    const auto dtype =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        dtype);

    return GRAPH_SUCCESS;
}

}  // namespace ge


namespace ops {

class Mul : public OpDef {
public:
    explicit Mul(const char *name)
        : OpDef(name)
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
            .SetInferDataType(
                ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops