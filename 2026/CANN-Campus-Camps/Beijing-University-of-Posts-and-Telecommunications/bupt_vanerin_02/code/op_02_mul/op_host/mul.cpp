// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    // EleWise 标准 tiling:
    //   多核切分: 每核至少 4KB(MIN_TILING_BITS), blockFormer 按 512 元素对齐,
    //             对齐后虚拟 block 数 blockNum 可能小于 coreNum, SetBlockDim 用 blockNum;
    //   UB 切分:  ubFormer 按 256B 对齐, 预算 = UB容量 / (双缓冲2份 x 3个tensor x 元素字节),
    //             并区分 普通block/末block 的循环次数与尾 tile 大小写回 tiling 结构体。
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 平台信息: 可用矢量核数 与 单核 UB 容量
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 输入信息(按算子约束 x/y/z 的 shape、dtype 一致, 取 x 即可)
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int64_t elem_bytes = ge::GetSizeByDataType(dtype_x); // fp32=4, fp16=2
        int64_t dim0 = tensor_x->GetShapeSize();             // 元素总数, 本算例 8*2048=16384

        // 配置 tiling key: 按 x 的 dtype 选择 kernel 模板实例 (DT_X = float/half)
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // ---- 多核切分 ----
        constexpr int64_t MIN_TILING_BITS = 65536;   // 每核至少 4KB 数据(单位 bit)
        constexpr int64_t ELEM_ALIGN_FACTOR = 512;   // 多核切分元素对齐因子
        constexpr int64_t UB_ALIGN_BYTES = 256;      // UB 处理量按 256B 对齐(矢量指令效率)
        constexpr int64_t TENSOR_NUM = 3;            // x, y, z 三个 tensor

        // 保证每核 >= 4KB 的前提下尽量多核
        int64_t coreNum = 8;
        coreNum = coreNum > num_cores_aiv ? num_cores_aiv : coreNum;
        coreNum = coreNum < 1 ? 1 : coreNum;

        // 每核基础元素数, 向上按 512 元素对齐
        int64_t blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1) /
                              ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
        // 对齐后实际需要的 block 数(可能 < coreNum, 启动核数以此为准)
        int64_t blockNum = (dim0 + blockFormer - 1) / blockFormer;

        // ---- UB 切分 ----
        // 双缓冲 x/y/z 共 MUL_BUFFER_NUM * TENSOR_NUM 份 tile 需同时驻留 UB
        int64_t maxElemNum = static_cast<int64_t>(ub_size) / (MUL_BUFFER_NUM * TENSOR_NUM * elem_bytes);
        int64_t alignFactor = UB_ALIGN_BYTES / elem_bytes; // fp32=64, fp16=128
        int64_t ubFormer = (maxElemNum / alignFactor) * alignFactor;
        if (ubFormer < alignFactor) {
            ubFormer = alignFactor; // UB 极小场景保底一个对齐单元
        }
        if (ubFormer > blockFormer) {
            ubFormer = blockFormer; // tile 大于单核任务量无意义(blockFormer 本身满足 256B 对齐)
        }

        // 普通 block / 末 block 的 UB 循环次数与尾 tile 大小
        int64_t ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
        int64_t ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;
        int64_t blockTail = dim0 - (blockNum - 1) * blockFormer;
        int64_t ubLoopOfTailBlock = (blockTail + ubFormer - 1) / ubFormer;
        int64_t ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;

        // 写回 tiling 结构体
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->dim0 = static_cast<uint32_t>(dim0);
        tiling->blockNum = static_cast<uint32_t>(blockNum);
        tiling->blockFormer = static_cast<uint32_t>(blockFormer);
        tiling->ubFormer = static_cast<uint32_t>(ubFormer);
        tiling->ubLoopOfFormerBlock = static_cast<uint32_t>(ubLoopOfFormerBlock);
        tiling->ubTailOfFormerBlock = static_cast<uint32_t>(ubTailOfFormerBlock);
        tiling->ubLoopOfTailBlock = static_cast<uint32_t>(ubLoopOfTailBlock);
        tiling->ubTailOfTailBlock = static_cast<uint32_t>(ubTailOfTailBlock);

        // 配置启动核数(用对齐后的 blockNum)与 workspace(本算子不需要)
        context->SetBlockDim(static_cast<uint32_t>(blockNum));
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    // 输出 z 与输入 x 形状一致(逐元素算子, 形状直通)
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *shape_x = context->GetInputShape(0);
        gert::Shape *shape_z = context->GetOutputShape(0);
        if (shape_x == nullptr || shape_z == nullptr) {
            return GRAPH_FAILED;
        }
        *shape_z = *shape_x;
        return GRAPH_SUCCESS;
    }
    // 输出 z 的 dtype 与输入 x 一致
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
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
