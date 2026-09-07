// Host侧 —— TilingFunc + InferShape + InferDataType
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // --- 平台信息 ---
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // --- 输入张量信息 ---
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();       // 数据类型
        uint32_t length_input_x = tensor_input_x->GetShapeSize();          // 元素个数

        // --- tiling key：根据 dtype 分发到不同模板实例 ---
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // --- 计算 blockDim：让每个核分到的 perCore 是 32B 对齐的，杜绝最后一块尾巴越界 ---
        int dtypeSize = ge::GetSizeByDataType(dtype_input_x);
        int32_t alignElements = 32 / dtypeSize;                    // fp32=8, fp16=16
        int32_t perCoreRaw = (int32_t)((length_input_x + num_cores_aiv - 1) / num_cores_aiv);
        int32_t perCoreAlign = ((perCoreRaw + alignElements - 1) / alignElements) * alignElements;
        int32_t blockDimNew = (int32_t)((length_input_x + perCoreAlign - 1) / perCoreAlign);
        if (blockDimNew < 1) blockDimNew = 1;
        if (blockDimNew > num_cores_aiv) blockDimNew = num_cores_aiv;

        // --- 写回 tiling 结构体（perCore 传下去，kernel 侧免除法） ---
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length_input_x;
        tiling->perCore = (uint32_t)perCoreAlign;

        // --- 启动核数（每核 perCore 对齐，kernel 侧用 GetBlockNum 算出一致值） ---
        context->SetBlockDim(blockDimNew);

        // --- workspace（本算子不需要） ---
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    // InferShape：output 的 shape 与 input_x 一致
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const auto *x_shape = context->GetInputShape(0);    // const gert::Shape*
        auto *y_shape = context->GetOutputShape(0);         // gert::Shape*
        if (x_shape == nullptr || y_shape == nullptr) {
            return GRAPH_FAILED;
        }
        *y_shape = *x_shape;                                // 值拷贝
        return GRAPH_SUCCESS;
    }

    // InferDataType：output 的 dtype 与 input_x 一致
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        ge::DataType x_dtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, x_dtype);
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Gelu : public OpDef {
    public:
        explicit Gelu(const char *name) : OpDef(name) {
            this->Input("input_x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Gelu);
}  // namespace ops
