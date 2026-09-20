// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (num_cores_aiv <= 0) {
            num_cores_aiv = 1;
        }

        // 获取输入信息
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        if (tensor_input_x == nullptr) {
            return ge::GRAPH_FAILED;
        }
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        int dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);
        if (dtype_size_input_x <= 0) {
            dtype_size_input_x = 4;
        }
        uint32_t length_input_x = static_cast<uint32_t>(tensor_input_x->GetShapeSize());

        // 配置tiling key, 区分 float / float16
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // 32B 对齐的单元（元素个数）
        int64_t unit = 32 / dtype_size_input_x;
        if (unit <= 0) {
            unit = 8;
        }

        // 多核切分：每核处理 blockFactor 个元素
        int64_t blockFactor = (static_cast<int64_t>(length_input_x) + num_cores_aiv - 1) / num_cores_aiv;
        blockFactor = (blockFactor + unit - 1) / unit * unit;
        if (blockFactor < unit) {
            blockFactor = unit;
        }
        int64_t usedCoreNum = (static_cast<int64_t>(length_input_x) + blockFactor - 1) / blockFactor;
        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }
        if (usedCoreNum > num_cores_aiv) {
            usedCoreNum = num_cores_aiv;
        }

        // UB 切分：预留 8KB 系统空间；输入/输出/两个临时 buffer 共 4 份
        uint64_t usable_ub_size = (ub_size > 8192) ? (ub_size - 8192) : ub_size;
        int64_t oneBufferElems = static_cast<int64_t>(usable_ub_size) / dtype_size_input_x / 4;
        int64_t ubFactor = oneBufferElems / unit * unit;
        if (ubFactor <= 0) {
            ubFactor = unit;
        }

        // 填充 tiling 结构体
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length_input_x;
        tiling->blockFactor = static_cast<uint32_t>(blockFactor);
        tiling->ubFactor = static_cast<uint32_t>(ubFactor);

        // 配置启动核数
        context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace != nullptr) {
            currentWorkspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        if (inputShape == nullptr) {
            return GRAPH_FAILED;
        }
        gert::Shape *outputShape = context->GetOutputShape(0);
        if (outputShape == nullptr) {
            return GRAPH_FAILED;
        }
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        ge::DataType inputDtype = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDtype);
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

