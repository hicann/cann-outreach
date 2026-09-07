// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 平台信息：AIV 核数与单核 UB 容量
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        // 输入信息：dtype 决定模板实例与元素字长，shape 决定切分
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);   // fp16=2B, fp32=4B
        uint32_t length_x = tensor_x->GetShapeSize();        // 元素个数，本题 16384

        // tiling key：按 dtype 选择 kernel 模板实例（float / half）
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        constexpr uint32_t BUFFER_NUM = 2;                   // 与 kernel 侧保持一致
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = length_x;

        if (length_x == 0) {                                 // 空输入防御
            tiling->blockLength = 0;
            tiling->tileNum = 0;
            tiling->tileLength = 0;
            context->SetBlockDim(1);
            size_t *ws0 = context->GetWorkspaceSizes(1);
            ws0[0] = 0;
            return ge::GRAPH_SUCCESS;
        }

        // ---- 核间切分 ----
        // 约束：blockDim 整除 totalLength，且每核块是 2*align 的倍数（2*align 保证
        //       核内还能对半切成双缓冲两轮；align = 32B 对齐元素数，fp16=16, fp32=8）。
        //       不满足则向下搜索，从构造上消灭尾块与错位窗口。
        uint32_t alignElems = 32 / dtype_size_x;
        uint32_t blockDim = static_cast<uint32_t>(num_cores_aiv);
        while (blockDim > 1 &&
               (length_x % blockDim != 0 ||
                (length_x / blockDim) % (2 * alignElems) != 0)) {
            blockDim--;
        }
        uint32_t blockLength = length_x / blockDim;          // 循环退出时保证整除

        // ---- 核内切分 ----
        // UB 预算：3 个队列(x/y/z) × BUFFER_NUM 份 × 每元素字长 ≤ ub_size
        uint32_t maxTile = static_cast<uint32_t>(ub_size / (3 * BUFFER_NUM * dtype_size_x));
        maxTile -= maxTile % (2 * alignElems);               // 对齐收紧，保证可整分
        // 默认每核 2 轮（tileNum=1 × BUFFER_NUM=2），每轮 blockLength/2 个元素
        uint32_t tileLength = blockLength / 2;
        uint32_t tileNum = 1;
        if (tileLength > maxTile) {                          // 单轮装不下 UB → 增加轮数
            uint32_t rounds = (blockLength + maxTile - 1) / maxTile;
            rounds += rounds & 1;                            // 取偶，配合双缓冲
            tileNum = rounds / 2;
            // 整除防御：向小回退到能整除 blockLength 的组数
            while (tileNum > 1 && blockLength % (tileNum * BUFFER_NUM) != 0) {
                tileNum--;
            }
            tileLength = blockLength / (tileNum * BUFFER_NUM);
        }

        tiling->blockLength = blockLength;
        tiling->tileNum = tileNum;
        tiling->tileLength = tileLength;

        // 本题实测（16384 元素，AIV=40）：blockDim 收敛到 32，blockLength=512，
        // tileNum=1，tileLength=256 → fp16 单核 UB 占用 3*2*256*2B = 3KB ≪ UB 容量
        context->SetBlockDim(blockDim);

        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // z 的 shape 与 x 一致（elementwise 算子，y 同形）
        const gert::Shape *xShape = context->GetInputShape(0);
        gert::Shape *zShape = context->GetOutputShape(0);
        if (xShape == nullptr || zShape == nullptr) {
            return GRAPH_FAILED;
        }
        *zShape = *xShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // z 的 dtype 与 x 一致（fp32/fp16 同精度运算，不发生类型提升）
        ge::DataType dtypeX = context->GetInputDataType(0);
        context->SetOutputDataType(0, dtypeX);
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
