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

    (void)ub_size;

    // 获取输入 Tensor
    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);

    (void)tensor_y;

    // 获取数据类型
    ge::DataType dtype_x =
        tensor_x->GetDataType();

    // 获取输入元素个数
    uint32_t length_x =
        tensor_x->GetShapeSize();

    // 获取输入内存大小
    uint32_t size_x =
        tensor_x->GetSize();

    (void)size_x;

    // =========================================================
    // 配置 Tiling Key
    // =========================================================
    uint32_t DT_X;

    if (dtype_x == ge::DT_FLOAT) {
        DT_X = C_DT_FLOAT;
    } else if (dtype_x == ge::DT_FLOAT16) {
        DT_X = C_DT_FLOAT16;
    } else {
        return ge::GRAPH_FAILED;
    }

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // =========================================================
    // 填充 TilingData
    // =========================================================
    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->length = length_x;

    // =========================================================
    // 配置启动核数
    //
    // Kernel中：
    //
    // blockLength = length / blockNum
    // tileLength  = 256
    //
    // 因此要求：
    //
    // length % blockNum == 0
    // blockLength % 256 == 0
    // =========================================================
    uint32_t block_dim =
        static_cast<uint32_t>(num_cores_aiv);

    if (block_dim > length_x) {
        block_dim = length_x;
    }

    while (block_dim > 1) {
        if (length_x % block_dim == 0) {
            uint32_t block_length =
                length_x / block_dim;

            if (block_length % 256 == 0) {
                break;
            }
        }

        --block_dim;
    }

    if (block_dim == 0) {
        block_dim = 1;
    }

    context->SetBlockDim(block_dim);

    // =========================================================
    // Workspace
    // =========================================================
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
    // 获取输入 x 和输出 z 的 Shape
    const gert::Shape *shape_x =
        context->GetInputShape(0);

    gert::Shape *shape_z =
        context->GetOutputShape(0);

    // z 的 Shape 与 x 保持一致
    *shape_z = *shape_x;

    return GRAPH_SUCCESS;
}


static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    // z 的 dtype 与 x 保持一致
    ge::DataType dtype_x =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        dtype_x);

    return GRAPH_SUCCESS;
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