// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    constexpr uint32_t BUFFER_NUM = 2;       // 与kernel侧队列深度保持一致
    constexpr uint32_t TILE_NUM = 8;         // 每个核内的分块数
    constexpr uint32_t MIN_TILE_LENGTH = 32; // 每块最少32个元素: fp16时64B、fp32时128B, 均满足32B对齐

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();

        // 获取算子输入信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        uint32_t totalLength = tensor_x->GetShapeSize();

        // 核间切分: 取totalLength的不超过AIV核数的最大因子, 保证均分无尾块
        uint32_t coreNum = static_cast<uint32_t>(num_cores_aiv);
        uint32_t blockDim = (coreNum < totalLength) ? coreNum : totalLength;
        while (blockDim > 1 && totalLength % blockDim != 0) {
            blockDim--;
        }
        uint32_t blockLength = totalLength / blockDim; // 每个核处理的元素个数

        // 核内切分: tileNum个分块, 每块由BUFFER_NUM个buffer轮转搬运
        uint32_t tileNum = TILE_NUM;
        while (tileNum > 1 && blockLength / (tileNum * BUFFER_NUM) < MIN_TILE_LENGTH) {
            tileNum /= 2;
        }

        // 配置tiling key, 实现kernel侧不同数据类型的区分
        uint32_t DT_X = static_cast<uint32_t>(tensor_x->GetDataType());
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 计算tiling方案并填充tiling结构体
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->totalLength = totalLength;
        tiling->tileNum = tileNum;

        // 配置启动核数
        context->SetBlockDim(blockDim);
        // 配置workspace大小
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 逐元素乘法: 输出z的shape与输入x一致
        const gert::Shape *x_shape = context->GetInputShape(0);
        gert::Shape *z_shape = context->GetOutputShape(0);
        *z_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 逐元素乘法: 输出z的dtype与输入x一致
        const ge::DataType inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
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
