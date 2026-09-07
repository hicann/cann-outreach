// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    // kernel 侧每个元素占用的 UB 字节数:
    // fp32: 双缓冲输入/输出 2*2*4B + abs/erfIn/erfOut 3*4B + Erf临时buffer 3*4B = 40B
    // fp16: 双缓冲输入/输出 2*2*2B + xFp32/yFp32/abs/erfIn/erfOut 5*4B + Erf临时buffer 3*4B = 40B
    constexpr uint32_t GELU_PER_ELEM_UB_BYTES = 40;
    constexpr uint32_t GELU_MAX_TILE_LENGTH = 8192;
    constexpr uint32_t GELU_UB_RESERVE_BYTES = 32 * 1024; // 预留栈/系统开销

    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t numCoresAiv = platform.GetCoreNumAiv();
        uint64_t ubSize = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

        const gert::Tensor *tensorInputX = context->GetRequiredInputTensor(0);
        ge::DataType dtypeInputX = tensorInputX->GetDataType();
        if (dtypeInputX != ge::DT_FLOAT && dtypeInputX != ge::DT_FLOAT16) {
            return ge::GRAPH_FAILED;
        }
        uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeInputX));
        uint64_t totalLength = static_cast<uint64_t>(tensorInputX->GetShapeSize());
        uint32_t alignElems = 32u / dtypeSize; // 32B 对齐所需的元素个数

        // UB 单 tile 长度: 按可用 UB 向下对齐到 32B, 并设置上限
        uint64_t usableUb = ubSize > GELU_UB_RESERVE_BYTES ? ubSize - GELU_UB_RESERVE_BYTES : ubSize;
        uint32_t tileLength = static_cast<uint32_t>(usableUb / GELU_PER_ELEM_UB_BYTES);
        tileLength = tileLength / alignElems * alignElems;
        tileLength = tileLength > GELU_MAX_TILE_LENGTH ? GELU_MAX_TILE_LENGTH : tileLength;
        tileLength = tileLength < alignElems ? alignElems : tileLength;

        // 多核切分: 按 32B 对齐的元素块均分, 尾核处理剩余部分
        uint32_t blockLength = 0;
        uint32_t usedCoreNum = 1;
        if (totalLength > 0 && numCoresAiv > 0) {
            uint64_t block = (totalLength + numCoresAiv - 1) / numCoresAiv;
            block = (block + alignElems - 1) / alignElems * alignElems;
            blockLength = static_cast<uint32_t>(block);
            usedCoreNum = static_cast<uint32_t>((totalLength + block - 1) / block);
        }

        uint32_t DT_INPUT_X = static_cast<uint32_t>(dtypeInputX);
        ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

        GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
        tiling->totalLength = totalLength;
        tiling->usedCoreNum = usedCoreNum;
        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;

        context->SetBlockDim(usedCoreNum);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        // 逐元素算子, 输出 shape 与输入相同
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        if (inputShape == nullptr || outputShape == nullptr) {
            return GRAPH_FAILED;
        }
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        // 输出 dtype 与输入 dtype 相同
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
