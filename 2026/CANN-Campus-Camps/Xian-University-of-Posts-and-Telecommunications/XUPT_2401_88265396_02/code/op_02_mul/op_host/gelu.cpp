// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    // 获取输入
    const gert::Tensor *tensor_input_x =
        context->GetRequiredInputTensor(0);

    // 输入数据类型
    ge::DataType dtype_input_x =
        tensor_input_x->GetDataType();

    // 输入总元素数
    uint32_t length_input_x =
        tensor_input_x->GetShapeSize();

    // ==============================
    // 配置 TilingKey
    // ==============================

    uint32_t DT_INPUT_X =
        static_cast<uint32_t>(
            dtype_input_x);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_INPUT_X);

    // ==============================
    // 填充 TilingData
    // ==============================

    GeluTilingData *tiling =
        context->GetTilingData<
            GeluTilingData>();

    tiling->length =
        length_input_x;

    /*
     * 为了保证不同测试 Shape 都能正确处理，
     * 先使用单 Block。
     *
     * Kernel 内部再进行分 Tile。
     */
    context->SetBlockDim(1);

    // 不需要 workspace
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    /*
     * GELU 为 element-wise 算子：
     *
     * output.shape = input.shape
     */
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
    /*
     * output.dtype = input.dtype
     */
    ge::DataType inputType =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        inputType);

    return GRAPH_SUCCESS;
}

} // namespace ge


namespace ops {

class Gelu : public OpDef {
public:
    explicit Gelu(const char *name)
        : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->SetInferShape(
                ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);

        this->AICore()
            .SetTiling(
                optiling::TilingFunc)
            .AddConfig(
                "ascend910b");
    }
};

OP_ADD(Gelu);

} // namespace ops