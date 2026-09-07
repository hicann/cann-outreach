// Host侧Tiling实现

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    uint64_t ub_size = 0;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ub_size);

    // 获取输入 Tensor
    const gert::Tensor *tensor_input_x =
        context->GetRequiredInputTensor(0);

    ge::DataType dtype_input_x =
        tensor_input_x->GetDataType();

    int dtype_size_input_x =
        ge::GetSizeByDataType(dtype_input_x);

    uint32_t length_input_x =
        tensor_input_x->GetShapeSize();

    // 数据类型模板选择
    uint32_t DT_INPUT_X =
        static_cast<uint32_t>(dtype_input_x);

    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    // ============================
    // Tiling 参数
    // ============================

    GeluTilingData *tiling =
        context->GetTilingData<GeluTilingData>();

    tiling->length = length_input_x;

    // 一个比较保守的 Tile 大小
    // 1024 对 float16 / float32 都满足32字节对齐
    tiling->tileLength = 1024;

    // 当前 Kernel 是单核处理整个 Tensor，
    // 所以这里必须设置为1，避免多个核重复计算。
    context->SetBlockDim(1);

    // 当前 Erf 使用 sharedTmpBuffer，
    // 不需要 Host workspace。
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


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

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(
        0,
        context->GetInputDataType(0));

    return GRAPH_SUCCESS;
}

}  // namespace ge


namespace ops {

class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name)
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

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Gelu);

}  // namespace ops