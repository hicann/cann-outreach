#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    // ============ 性能可调旋钮 ============
    const uint32_t BUFFER_NUM = 2;      // 与 Kernel 保持一致
    const int32_t BLOCK_DIM_CAP = 16;   // 核数封顶
    const uint32_t UB_RESERVE = 8192;   // UB 预留
    const uint32_t UB_SPLIT = 1;        // 切块系数（1=单次搬完，>1 强制切块）
    // =====================================

    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_input_x = tensor_input_x->GetDataType();
        int dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);
        uint32_t length_input_x = tensor_input_x->GetShapeSize();

        // 设置 Tiling Key，用于 Kernel 模板实例化
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();

        uint32_t total = length_input_x;
        if (total < 1) total = 1;
        if (dtype_size_input_x < 1) dtype_size_input_x = 4;
        if (ub_size == 0) ub_size = 192 * 1024; // 910B 默认 192KB

        // ---- 核数：min(平台核数, 封顶, 总元素数) ----
        uint32_t blockDim = 1;
        if (num_cores_aiv > 0) blockDim = static_cast<uint32_t>(num_cores_aiv);
        if (blockDim > static_cast<uint32_t>(BLOCK_DIM_CAP)) blockDim = BLOCK_DIM_CAP;
        if (blockDim > total) blockDim = total;

        // ---- 每核元素数（向上取整） ----
        uint32_t blockLength = (total + blockDim - 1) / blockDim;

        // ---- UB 允许的最大分块 ----
        // 队列占用：2 * BUFFER_NUM * tile * dtypeSize
        // float 工作区：3 * tile * 4
        uint32_t bytesPerElem = static_cast<uint32_t>(2 * BUFFER_NUM * dtype_size_input_x) + 3 * 4;
        uint64_t usable = (ub_size > UB_RESERVE) ? (ub_size - UB_RESERVE) : ub_size;
        uint32_t maxTile = static_cast<uint32_t>(usable / bytesPerElem);
        if (maxTile < 1) maxTile = 1;

        uint32_t tileLength = blockLength;
        if (tileLength > maxTile) tileLength = maxTile;

        // 若 UB_SPLIT > 1 且数据量允许，强制切块以利用双缓冲
        if (UB_SPLIT > 1 && blockLength > 1) {
            uint32_t split = (blockLength + UB_SPLIT - 1) / UB_SPLIT;
            if (split > maxTile) split = maxTile;
            if (split < 1) split = 1;
            if (split < tileLength) tileLength = split;
        }

        tiling->totalLength = total;
        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;

        context->SetBlockDim(blockDim);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace != nullptr) currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
} // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        if (xShape == nullptr || yShape == nullptr) return GRAPH_FAILED;
        *yShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
} // namespace ge

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
} // namespace ops