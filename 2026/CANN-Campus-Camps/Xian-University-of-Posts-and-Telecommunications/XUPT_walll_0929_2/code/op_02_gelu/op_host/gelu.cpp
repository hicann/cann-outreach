// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        const auto *input = context->GetRequiredInputTensor(0);
        const auto dtype = input->GetDataType();
        if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
            return ge::GRAPH_FAILED;
        }
        uint32_t DT_X = static_cast<uint32_t>(dtype);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        const auto &shape = context->GetInputShape(0)->GetStorageShape();
        uint64_t totalLength = 1;
        for (size_t i = 0; i < shape.GetDimNum(); ++i) {
            const int64_t dim = shape.GetDim(i);
            if (dim < 0 || (dim != 0 && totalLength > UINT32_MAX / static_cast<uint64_t>(dim))) {
                return ge::GRAPH_FAILED;
            }
            totalLength *= static_cast<uint64_t>(dim);
        }

        constexpr uint32_t tileLength = 256;
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = platform.GetCoreNumAiv();
        if (coreNum == 0) {
            return ge::GRAPH_FAILED;
        }
        uint32_t blockDim = coreNum < 8 ? coreNum : 8;
        uint32_t tileCount = static_cast<uint32_t>((totalLength + tileLength - 1) / tileLength);
        if (tileCount == 0) {
            tileCount = 1;
        }
        if (blockDim > tileCount) {
            blockDim = tileCount;
        }

        auto *tiling = context->GetTilingData<GeluTilingData>();
        tiling->totalLength = static_cast<uint32_t>(totalLength);
        tiling->tileLength = tileLength;
        context->SetBlockDim(blockDim);
        context->GetWorkspaceSizes(1)[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        *context->GetOutputShape(0) = *context->GetInputShape(0);
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
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
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("output")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Gelu);
}  // namespace ops
