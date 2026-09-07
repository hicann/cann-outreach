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
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 获取算子输入数组信息
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        int dtype_size = ge::GetSizeByDataType(dtype_input_x);  // fp16=2, fp32=4
        uint32_t total_length = tensor_input_x->GetShapeSize();

        // 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
         uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // 一个 32 字节数据块包含的元素个数
        uint32_t align_num = 32 / dtype_size;

        // 估算每个 tile 在 UB 中的占用量（含余量），得到 32 字节对齐的 tile 长度
        uint32_t per_elem_budget = 32;
        uint32_t tile_length = (ub_size / per_elem_budget) / align_num * align_num;
        if (tile_length < align_num) {
            tile_length = align_num;
        }

        // 计算总 tile 数并确定实际启动的核数
        uint32_t total_tiles = (total_length + tile_length - 1) / tile_length;
        uint32_t block_dim = static_cast<uint32_t>(num_cores_aiv);
        if (block_dim > total_tiles) {
            block_dim = total_tiles;
        }
        if (block_dim == 0) {
            block_dim = 1;
        }

        // 填充 tiling 结构体
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->totalLength = total_length;
        tiling->tileLength = tile_length;
        tiling->alignNum = align_num;

        // 配置启动核数
        context->SetBlockDim(block_dim);
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
