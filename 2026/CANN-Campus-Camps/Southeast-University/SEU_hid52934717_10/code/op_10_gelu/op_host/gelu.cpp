// Host侧Tiling实现
#include <algorithm>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 示例: 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 示例: 获取算子输入数组信息
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType(); // 获取数据类型
        int dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x); // 获取数据类型的字长
        uint32_t length_input_x = tensor_input_x->GetShapeSize(); // 获取元素个数
        uint32_t size_input_x = tensor_input_x->GetSize(); // 获取内存大小
        // 示例: 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);
        // 示例: 计算tiling方案并填充tiling结构体
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length_input_x;
        // 配置启动核数
        // The kernel assigns independent tiles to AIVs; queue synchronization
        // keeps each core's GM/vector pipeline ordered.
        const uint32_t tile_count = (length_input_x + 1024 - 1) / 1024;
        context->SetBlockDim(std::min<uint32_t>(num_cores_aiv, tile_count));
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *input_shape = context->GetInputShape(0);
        gert::Shape *output_shape = context->GetOutputShape(0);
        if (input_shape == nullptr || output_shape == nullptr) {
            return GRAPH_FAILED;
        }
        *output_shape = *input_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
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