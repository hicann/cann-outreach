// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t numCoresAiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
        MulTilingData *tiling = context->GetTilingData<MulTilingData>();
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (tensorX == nullptr || tiling == nullptr || currentWorkspace == nullptr || numCoresAiv == 0 || ub_size == 0) {
            return ge::GRAPH_FAILED;
        }

        const ge::DataType dtypeX = tensorX->GetDataType();
        const int32_t dtypeSize = ge::GetSizeByDataType(dtypeX);
        if (dtypeSize <= 0) {
            return ge::GRAPH_FAILED;
        }
        ASCENDC_TPL_SEL_PARAM(context, static_cast<uint32_t>(dtypeX));

        constexpr uint32_t kDmaAlignmentBytes = 32;
        constexpr uint32_t kMinBytesPerCore = 4096;
        constexpr uint32_t kQueueCount = 3;
        constexpr uint32_t kBufferCount = 2;
        const uint32_t length = tensorX->GetShapeSize();
        const uint32_t elementsPerDmaBlock = kDmaAlignmentBytes / static_cast<uint32_t>(dtypeSize);
        if (elementsPerDmaBlock == 0 || ub_size < kQueueCount * kBufferCount * kDmaAlignmentBytes) {
            return ge::GRAPH_FAILED;
        }

        tiling->length = length;
        tiling->blockLength = 0;
        tiling->tileLength = 0;
        currentWorkspace[0] = 0;
        if (length == 0) {
            context->SetBlockDim(1);
            return ge::GRAPH_SUCCESS;
        }

        const uint64_t totalBytes = static_cast<uint64_t>(length) * static_cast<uint32_t>(dtypeSize);
        uint32_t blockDim = static_cast<uint32_t>((totalBytes + kMinBytesPerCore - 1) / kMinBytesPerCore);
        blockDim = (blockDim < numCoresAiv) ? blockDim : numCoresAiv;
        blockDim = (blockDim == 0) ? 1 : blockDim;

        const uint32_t elementsPerCore = (length + blockDim - 1) / blockDim;
        const uint32_t blockLength =
            ((elementsPerCore + elementsPerDmaBlock - 1) / elementsPerDmaBlock) * elementsPerDmaBlock;
        const uint32_t activeCores = (length + blockLength - 1) / blockLength;

        uint64_t tileBytes = ub_size / (kQueueCount * kBufferCount);
        tileBytes = (tileBytes / kDmaAlignmentBytes) * kDmaAlignmentBytes;
        const uint32_t tileLength = static_cast<uint32_t>(tileBytes / static_cast<uint32_t>(dtypeSize));
        if (tileLength == 0) {
            return ge::GRAPH_FAILED;
        }

        tiling->blockLength = blockLength;
        tiling->tileLength = tileLength;
        context->SetBlockDim(activeCores);
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        if (inputShape == nullptr || outputShape == nullptr) {
            return GRAPH_FAILED;
        }
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
}  // namespace ops
