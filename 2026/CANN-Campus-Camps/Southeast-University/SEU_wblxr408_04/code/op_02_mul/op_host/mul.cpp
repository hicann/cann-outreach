// Host-side tiling implementation.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace {
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BUFFER_COUNT = 3;

uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    return factor == 0 ? 0 : (value + factor - 1) / factor;
}

uint32_t AlignUp(uint32_t value, uint32_t align)
{
    return align == 0 ? value : CeilDiv(value, align) * align;
}

uint32_t AlignDown(uint32_t value, uint32_t align)
{
    return align == 0 ? value : value / align * align;
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t coreNumAiv = platform.GetCoreNumAiv();
    uint32_t coreNum = coreNumAiv > 0 ? static_cast<uint32_t>(coreNumAiv) : 1;

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    ge::DataType dtypeX = tensorX->GetDataType();
    uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeX));
    uint32_t totalLength = static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    uint32_t alignNum = BYTE_BLOCK / dtypeSize;
    uint32_t blockLength = AlignUp(CeilDiv(totalLength, coreNum), alignNum);
    if (blockLength == 0) {
        blockLength = alignNum;
    }

    uint32_t blockDim = CeilDiv(totalLength, blockLength);
    if (blockDim == 0) {
        blockDim = 1;
    }

    uint32_t tileLength = static_cast<uint32_t>(ubSize / (BUFFER_NUM * BUFFER_COUNT * dtypeSize));
    tileLength = AlignDown(tileLength, alignNum);
    if (tileLength == 0) {
        tileLength = alignNum;
    }
    if (tileLength > blockLength) {
        tileLength = blockLength;
    }

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;

    context->SetBlockDim(blockDim);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name)
    {
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
