// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取输入信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        ge::DataType dtype_x = tensor_x->GetDataType();
        uint32_t length_x = tensor_x->GetShapeSize();

        // 配置 tiling key（根据数据类型选择）
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 计算 tiling 方案
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;

        // 每个核处理的元素数量，按 core 数均分
        uint32_t blockLength = (length_x + num_cores_aiv - 1) / num_cores_aiv;
        // 对齐到 8（float32/float16 都是 8 元素一组）
        blockLength = (blockLength + 7) / 8 * 8;
        tiling->blockLength = blockLength;

        // 配置启动核数
        context->SetBlockDim(num_cores_aiv);
        // 配置 workspace 大小（无需 workspace）
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static ge::graphStatus InferShapeCore(gert::InferShapeContext *context) {
        // 输出 z 的 shape 与输入 x 一致
        const gert::Shape *input_shape = context->GetInputShape(0);
        if (input_shape == nullptr) {
            return GRAPH_FAILED;
        }
        gert::Shape *output_shape = context->GetOutputShape(0);
        if (output_shape == nullptr) {
            return GRAPH_FAILED;
        }
        *output_shape = *input_shape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferShape(gert::InferShapeContext *context) {
        return InferShapeCore(context);
    }

    static graphStatus InferDataTypeCore(gert::InferDataTypeContext *context) {
        // 输出 z 的 dtype 与输入 x 一致
        ge::DataType input_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, input_dtype);
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return InferDataTypeCore(context);
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
