// ============================================================================
// op_02 Gelu 激活算子 —— Host 侧实现（Tiling + InferShape + InferDataType + OpDef）
// 功能：y = GELU(x)，x / y 均为 float32 / float16 张量，ND 布局
//
// v16 修改（对照他人通过的参考代码）：
//   - 启动核数使用平台全部 AIV 核（GetCoreNumAiv()）；
//   - 每核数据在 kernel 内按 32 元素对齐块分配（大核/小核），保证 GM 32 字节对齐；
//   - tile 长度由 UB 容量决定（双缓冲 2 * 输入输出 2 * dtype_size，留 3/4 UB），
//     下限 32、上限输入长度、再 32 对齐。
// ============================================================================
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 获取输入张量信息
        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        int32_t dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);  // fp16=2, fp32=4
        uint32_t length_input_x = tensor_input_x->GetShapeSize();            // 元素个数

        // 配置 tiling key（区分 fp32 / fp16 内核分支）
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        // 填充 tiling 结构体
        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->length = length_input_x;

        // 核数：使用平台全部 AIV 核，让每核数据量尽量小
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint32_t blockDim = (num_cores_aiv > 0) ? static_cast<uint32_t>(num_cores_aiv) : 8u;
        if (length_input_x < blockDim) {
            blockDim = length_input_x;
        }
        if (blockDim == 0) {
            blockDim = 1;
        }
        tiling->blockDim = blockDim;
        context->SetBlockDim(blockDim);

        // 每 tile 元素数：由 UB 容量决定（双缓冲 2 * 输入输出 2 * dtype_size，留 3/4 UB）
        // 下限 32，上限输入长度（32 对齐后），再 32 对齐
        uint32_t available_ub = static_cast<uint32_t>(ub_size) * 3 / 4;
        uint32_t tileLength = available_ub / (2 * 2 * static_cast<uint32_t>(dtype_size_input_x));
        if (tileLength < 32u) {
            tileLength = 32u;
        }
        uint32_t aligned_length = ((length_input_x + 31u) / 32u) * 32u;
        if (tileLength > aligned_length) {
            tileLength = aligned_length;
        }
        if (tileLength == 0) {
            tileLength = 32u;
        }
        tileLength = ((tileLength + 31u) / 32u) * 32u;
        tiling->tileLength = tileLength;

        // workspace 大小（本算子无需额外 workspace）
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        auto output_shape = context->GetOutputShape(0);
        *output_shape = *context->GetInputShape(0);
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Gelu : public OpDef {
    public:
        explicit Gelu(const char *name) : OpDef(name)
        {
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