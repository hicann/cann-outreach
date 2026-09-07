// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType(); // 获取数据类型
        uint32_t length_input_x = tensor_input_x->GetShapeSize(); // 获取元素个数
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length_input_x;
        uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
        if (num_cores_aiv <= 0) {
            block_dim = 1;
        }
        const uint32_t alignElements = 32 / ge::GetSizeByDataType(dtype_input_x);
        const uint32_t alignedBlocks =
            length_input_x / alignElements + (length_input_x % alignElements != 0);
        if (alignedBlocks < block_dim) {
            block_dim = alignedBlocks == 0 ? 1 : alignedBlocks;
        }
        const uint32_t blocksPerCore =
            alignedBlocks / block_dim + (alignedBlocks % block_dim != 0);
        tiling->blockLength = blocksPerCore * alignElements;
        tiling->tileLength = 4096;
        context->SetBlockDim(block_dim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
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
