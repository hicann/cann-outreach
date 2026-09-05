// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    // 单核内分块数量，与 KernelMul 中的 tileLength 计算保持一致。
    const uint32_t TILE_NUM = 8;
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取输入张量及其数据类型，并将类型作为模板参数传递给 AI Core。
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        // 计算输入张量的总元素个数，逐元素乘法要求两个输入形状一致。
        const gert::StorageShape* x1_shape = context->GetInputShape(0);
        int32_t data_sz = 1;
        for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
            data_sz *= x1_shape->GetStorageShape().GetDim(i);
        tiling->totalLength = static_cast<uint32_t>(data_sz);
        tiling->tileNum = TILE_NUM;

        // 配置 AI Core 启动数量，每个核处理一部分连续数据。
        context->SetBlockDim(8);  //num_cores_aiv
        // 本算子不需要额外的 workspace。
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 获取第0号输入 x 的 shape（静态/动态维度信息）
        const gert::Shape* x1_shape = context->GetInputShape(0);
        // 获取输出 z 的 shape，并复制输入 x 的 shape。
        gert::Shape* z_shape = context->GetOutputShape(0);
        // 逐元素乘法：输出 shape 与输入 shape 完全一致。
        *z_shape = *x1_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出 z 与输入 x 使用相同的数据类型。
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    // 注册矢量乘法算子：z = x * y。
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
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    // OP_ADD 是框架提供的通用 OpDef 注册宏，参数为当前算子类 Mul。
    OP_ADD(Mul);
}  // namespace ops
