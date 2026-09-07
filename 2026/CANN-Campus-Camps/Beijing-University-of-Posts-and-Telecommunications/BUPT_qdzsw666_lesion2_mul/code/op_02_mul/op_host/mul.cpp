// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 获取输入Tensor信息
    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);

    // 获取输入x的数据类型，并选择对应模板
    uint32_t DT_X =
        static_cast<uint32_t>(tensor_x->GetDataType());

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 获取Tiling数据空间
    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    // 获取输入x的存储形状
    const gert::StorageShape *x_shape =
        context->GetInputShape(0);

    // 计算输入Tensor的总元素数
    int32_t dataSize = 1;

    for (int32_t i = 0;
         i < x_shape->GetStorageShape().GetDimNum();
         i++) {
        dataSize *=
            x_shape->GetStorageShape().GetDim(i);
    }

    // 设置Tiling参数
    tiling->totalLength = dataSize;
    tiling->tileNum = TILE_NUM;

    // 使用8个AI Vector Core
    context->SetBlockDim(8);

    // 本算子不需要额外workspace
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    // 逐元素乘法：输出z的形状与输入x一致
    const gert::Shape *x_shape =
        context->GetInputShape(0);

    gert::Shape *z_shape =
        context->GetOutputShape(0);

    *z_shape = *x_shape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    // 输出z的数据类型与输入x一致
    const auto inputDataType =
        context->GetInputDataType(0);

    context->SetOutputDataType(0, inputDataType);

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