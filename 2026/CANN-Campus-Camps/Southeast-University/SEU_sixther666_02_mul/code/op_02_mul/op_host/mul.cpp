// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    constexpr uint32_t BLOCK_DIM = 8;

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 固定规格为(8, 2048)，每核连续处理一行。
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        if (tensor_x == nullptr) {
            return ge::GRAPH_FAILED;
        }
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint64_t length_x = tensor_x->GetShapeSize();
        if (length_x != 8U * 2048U) {
            return ge::GRAPH_FAILED;
        }

        // 配置tiling key，在kernel侧实例化匹配的float16/float32模板。
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 仅传递必要字段，降低小shape场景下的tiling数据搬运开销。
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = static_cast<uint32_t>(length_x);

        context->SetBlockDim(BLOCK_DIM);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        const auto input_data_type = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_data_type);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
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
    OP_ADD(Mul);
}  // namespace ops