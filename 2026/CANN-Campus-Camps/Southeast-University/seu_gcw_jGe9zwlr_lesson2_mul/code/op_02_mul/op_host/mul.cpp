// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 获取算子输入数组信息
    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);

    uint32_t DT_X =
        static_cast<uint32_t>(tensor_x->GetDataType());

    // 根据输入 dtype 选择 TilingKey
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 获取 TilingData
    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    // 获取输入 shape
    const gert::StorageShape* x1_shape =
        context->GetInputShape(0);

    // 计算输入元素总数
    int32_t data_sz = 1;

    for (int i = 0;
         i < x1_shape->GetStorageShape().GetDimNum();
         i++) {

        data_sz *=
            x1_shape->GetStorageShape().GetDim(i);
    }

    // 写入 Tiling 参数
    tiling->totalLength = data_sz;
    tiling->tileNum = TILE_NUM;

    // 配置启动核数
    context->SetBlockDim(8);

    // 配置 workspace 大小
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context) {

    // 获取输入 x 的 shape
    const gert::Shape* x1_shape =
        context->GetInputShape(0);

    // 获取输出 z
    gert::Shape* z_shape =
        context->GetOutputShape(0);

    // z 的 shape 与 x 一致
    *z_shape = *x1_shape;

    return GRAPH_SUCCESS;
}


static graphStatus InferDataType(
    gert::InferDataTypeContext *context) {

    // 输出 dtype 与输入 x 一致
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
    explicit Mul(const char *name) : OpDef(name) {

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