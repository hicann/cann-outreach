// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    constexpr uint32_t ONE_BLK_SIZE = 32;  // DataCopy 要求 32 字节对齐
    constexpr uint32_t BUFFER_NUM = 2;     // 与 kernel 侧队列缓冲个数保持一致
    constexpr uint32_t TENSOR_NUM = 3;     // x / y / z 三个队列

    // 每核最小元素数：限制核数上限，避免分块过小导致 DMA 效率下降。
    // V6：fp16 32 核 × 1KB 单块（上探并发流数）、fp32 32 核 × 2KB 单块（复验 V5 配置）。
    // V1~V5 实测模型：T ≈ 固定开销(~0.9us) + 流量/聚合带宽(随并发核数增长)。
    constexpr uint32_t MIN_BLOCK_ELEMS = 512;

    // 核数策略：0 表示自动（按 MIN_BLOCK_ELEMS 限制核数）。
    // 调优对比时可改为 8 / 16 / 32 等固定核数（如 CORE_NUM_OVERRIDE = 8）。
    constexpr uint32_t CORE_NUM_OVERRIDE = 0;

    static inline uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
    static inline uint32_t AlignUp(uint32_t a, uint32_t align) { return CeilDiv(a, align) * align; }
    static inline uint32_t AlignDown(uint32_t a, uint32_t align) { return a / align * align; }

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        // 获取算子输入信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();
        // 配置tiling key, 从而实现kernel侧不同数据类型/算法的区分
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 计算tiling方案并填充tiling结构体
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        if (length_x == 0) {
            // 空tensor场景：不启动计算循环，但需保证 InitBuffer 大小合法
            tiling->totalLength = 0;
            tiling->blockLength = ONE_BLK_SIZE;
            tiling->tileNum = 0;
            tiling->tileLength = ONE_BLK_SIZE;
            context->SetBlockDim(1);
            size_t *currentWorkspace = context->GetWorkspaceSizes(1);
            currentWorkspace[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // ===== 核间切分：每核块元素数不低于 MIN_BLOCK_ELEMS，块大小 32B 对齐，尾核收尾 =====
        // fp16 对齐单元 16 元素，fp32 对齐单元 8 元素
        uint32_t alignUnit = ONE_BLK_SIZE / static_cast<uint32_t>(dtype_size_x);
        uint32_t coreNum = (num_cores_aiv > 0) ? static_cast<uint32_t>(num_cores_aiv) : 1;
        if (CORE_NUM_OVERRIDE > 0 && CORE_NUM_OVERRIDE < coreNum) {
            coreNum = CORE_NUM_OVERRIDE;
        }
        uint32_t maxCoresByBlock = CeilDiv(length_x, MIN_BLOCK_ELEMS);
        if (coreNum > maxCoresByBlock) {
            coreNum = maxCoresByBlock;
        }
        uint32_t blockLength = AlignUp(CeilDiv(length_x, coreNum), alignUnit);
        uint32_t usedCoreNum = CeilDiv(length_x, blockLength);

        // ===== 核内切分：优先单块整搬（tileNum=1，减少 DMA 次数与队列开销）=====
        // 计算量极小（每核 ≤ 16 repeats），流水收益小于碎搬运损失；仅超大 shape 超 UB 预算时才分块。
        uint32_t maxTileLength = static_cast<uint32_t>(ub_size / (BUFFER_NUM * TENSOR_NUM * dtype_size_x));
        uint32_t tileLength = blockLength;
        if (tileLength > maxTileLength) {
            tileLength = AlignDown(maxTileLength, alignUnit);
            if (tileLength == 0) {
                tileLength = alignUnit;
            }
        }
        uint32_t tileNum = CeilDiv(blockLength, tileLength);

        tiling->totalLength = length_x;
        tiling->blockLength = blockLength;
        tiling->tileNum = tileNum;
        tiling->tileLength = tileLength;

        // 配置启动核数
        context->SetBlockDim(usedCoreNum);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 逐元素算子：输出 z 的 shape 与输入 x 一致
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出 z 的 dtype 与输入 x 一致
        const ge::DataType inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return GRAPH_SUCCESS;
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
