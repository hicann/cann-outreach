// Host 侧：多核版本，x/y 为同 shape、同 dtype 的连续张量。
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        const int64_t length_x = tensor_x->GetShapeSize();
        if (length_x < 0 || static_cast<uint64_t>(length_x) > UINT32_MAX) {
            return ge::GRAPH_FAILED;
        }

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        tiling->length = static_cast<uint32_t>(length_x);

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        const uint32_t coreNum = platform.GetCoreNumAiv();
        if (coreNum == 0) {
            return ge::GRAPH_FAILED;
        }
        // 与 Kernel 中的 TILE_LENGTH=1024 一致，最多每个 tile 使用一个核。
        const uint32_t tileNum = static_cast<uint32_t>((length_x + 1023) / 1024);
        const uint32_t blockDim = tileNum == 0 ? 1 : (tileNum < coreNum ? tileNum : coreNum);
        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
} // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }
} // namespace ge

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
} // namespace ops
