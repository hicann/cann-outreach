// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    // ============ 性能可调旋钮（改一个数即可 A/B，改完务必 clean rebuild）============
    const uint32_t BUFFER_NUM = 2;      // 与 op_kernel/gelu.cpp 保持一致
    const uint32_t MIN_ELEMS_PER_CORE = 256;  // 自适应下限：每核至少分摊的元素数，
                                              // 数据量极小时退化为 1 核（零跨核同步）
    const uint32_t BLOCK_DIM_CAP = 24;   // 中小 shape 核数封顶：延迟受限，核数越少跨核同步开销越小
    const uint32_t UB_RESERVE = 8192;   // UB 预留（队列元数据 + Erf 内部临时空间）
    const uint32_t UB_SPLIT = 1;        // 1=单核一次搬完(loopCount=1)，双缓冲无重叠收益，无需切块
    // 核数策略：默认封顶 BLOCK_DIM_CAP(8)；仅当 total > 8*maxTile（超大 shape、单核 loopCount
    // 过高）才放开到平台全部 AIV 核，用并行换吞吐。建议评测机实测 {1,8,16,30} 取最快。
    // =============================================================================

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

        // 配置tiling key，实现kernel侧不同数据类型的区分
        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();

        uint32_t total = length_input_x;
        if (total < 1) {
            total = 1;
        }
        if (dtype_size_input_x < 1) {
            dtype_size_input_x = 4;
        }
        if (ub_size == 0) {
            ub_size = 192 * 1024;  // 兜底：910B 约 192KB
        }

        // ---- UB 允许的最大分块（先算出来，用于按数据量选核数） ----
        // 队列占用 2 * BUFFER_NUM * tile * dtypeSize；float 工作区 3 * tile * 4
        uint32_t bytesPerElem = static_cast<uint32_t>(2 * BUFFER_NUM * dtype_size_input_x) + 3 * 4;
        uint64_t usable = (ub_size > UB_RESERVE) ? (ub_size - UB_RESERVE) : ub_size;
        uint32_t maxTile = static_cast<uint32_t>(usable / bytesPerElem);
        if (maxTile < 1) {
            maxTile = 1;
        }

        // ---- 核数：按数据量自适应封顶 ----
        // 默认封顶 8（与之前一致），中小 shape 仍由 MIN_ELEMS_PER_CORE 自然分配 1~8 核；
        // 只有超大 shape（单核 loopCount 过高）才放开到平台全部 AIV 核，用并行换吞吐。
        uint32_t blockDimCap = BLOCK_DIM_CAP;
        if (total > 8 * maxTile) {
            blockDimCap = (num_cores_aiv > 0) ? static_cast<uint32_t>(num_cores_aiv) : 1;
        }

        uint32_t blockDim = (total + MIN_ELEMS_PER_CORE - 1) / MIN_ELEMS_PER_CORE;
        if (blockDim < 1) {
            blockDim = 1;
        }
        if (num_cores_aiv > 0 && blockDim > static_cast<uint32_t>(num_cores_aiv)) {
            blockDim = static_cast<uint32_t>(num_cores_aiv);
        }
        if (blockDim > blockDimCap) {
            blockDim = blockDimCap;
        }
        if (blockDim > total) {
            blockDim = total;
        }

        // ---- 每核元素数：向上取整。无需 32B 对齐，非对齐由 kernel 的 DataCopyPad 处理 ----
        uint32_t blockLength = (total + blockDim - 1) / blockDim;

        uint32_t tileLength = blockLength;
        if (tileLength > maxTile) {
            tileLength = maxTile;
        }
        // 若单核能一次装下(loopCount==1)，双缓冲无从重叠；UB_SPLIT>1 时强制切块
        if (UB_SPLIT > 1 && blockLength > 1) {
            uint32_t split = (blockLength + UB_SPLIT - 1) / UB_SPLIT;
            if (split > maxTile) {
                split = maxTile;
            }
            if (split < 1) {
                split = 1;
            }
            if (split < tileLength) {
                tileLength = split;
            }
        }

        tiling->totalLength = total;
        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;

        context->SetBlockDim(blockDim);

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace != nullptr) {
            currentWorkspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    // 输出 shape 与输入完全一致（支持任意多维、非对齐 N）
    static graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *yShape = context->GetOutputShape(0);
        if (xShape == nullptr || yShape == nullptr) {
            return ge::GRAPH_FAILED;
        }
        *yShape = *xShape;
        return ge::GRAPH_SUCCESS;
    }
    // 输出 dtype 与输入一致（fp16 -> fp16, fp32 -> fp32）
    static graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
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
