// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"          // ★ 补：TypeUtils 在这

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"
constexpr uint32_t BUFFER_NUM = 2;   // 与 kernel 侧保持一致（双缓冲）

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());   // ← 变量名是 platform
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);             // ← 用 platform

        // 获取输入数组信息
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        const gert::Tensor *tensor_y = context->GetRequiredInputTensor(1);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();
        uint32_t size_x = tensor_x->GetSize();

        // 配置 tiling key
        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        // 填充 tiling：用 UB 推导单次最大搬运量（填满 UB，而不是写死 tileNum）
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = length_x;
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);   // ← 需 header
        uint32_t tileBlockNum = (uint32_t)(ub_size / 32u) / (3u*BUFFER_NUM);   // 3 = 2输入 + 1输出
        if (tileBlockNum == 0) tileBlockNum = 1;
        tiling->tileDataNum = tileBlockNum * 32u / typeLength;   // 填满UB的元素数

        // 选一个"能整除 length_x"的核数（保证每核切分不丢、对齐）
        int32_t blockNum = (num_cores_aiv > 0) ? num_cores_aiv : 1;
        if (blockNum > static_cast<int32_t>(length_x)) blockNum = static_cast<int32_t>(length_x);
        while (blockNum > 1 && (length_x % static_cast<uint32_t>(blockNum) != 0)) {
            blockNum--;
        }
        context->SetBlockDim(blockNum);

        // workspace（不需要）
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
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
}  // namespace op