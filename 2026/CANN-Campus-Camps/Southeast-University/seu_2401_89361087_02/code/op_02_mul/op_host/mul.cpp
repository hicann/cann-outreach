// Host侧Tiling实现

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"


namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 获取平台信息
    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    int32_t num_cores_aiv =
        platform.GetCoreNumAiv();

    uint64_t ub_size;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ub_size);

    // 获取输入Tensor
    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);

    // 获取输入数据类型
    ge::DataType dtype_x =
        tensor_x->GetDataType();

    int dtype_size_x =
        ge::GetSizeByDataType(dtype_x);

    // 获取x的元素个数
    uint32_t length_x =
        tensor_x->GetShapeSize();

    uint32_t size_x =
        tensor_x->GetSize();

    // 配置tiling key
    // 用于Kernel侧区分float / float16
    uint32_t DT_X =
        static_cast<uint32_t>(dtype_x);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_X);

    // 获取TilingData
    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    // 本题shape = (8, 2048)
    // 总元素数 = 16384
    tiling->totalLength =
        length_x;

    // 每个核划分8个tile
    tiling->tileNum = 8;

    // 本题使用8个Vector Core
    // 如果设备不足8核，则使用实际可用核数
    uint32_t blockDim = 8;

    if (num_cores_aiv < 8) {
        blockDim =
            static_cast<uint32_t>(
                num_cores_aiv);
    }

    context->SetBlockDim(
        blockDim);

    // 配置workspace大小
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


namespace ge {

// 输出z的shape与输入x保持一致
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


// 输出z的数据类型与输入x保持一致
static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    ge::DataType dtype =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        dtype);

    return ge::GRAPH_SUCCESS;
}

} // namespace ge


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

        // Shape推导和数据类型推导
        this->SetInferShape(
                ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);

        // 注册AI Core Kernel
        this->AICore()
            .SetTiling(
                optiling::TilingFunc)
            .AddConfig(
                "ascend910b");
    }
};

OP_ADD(Mul);

} // namespace ops