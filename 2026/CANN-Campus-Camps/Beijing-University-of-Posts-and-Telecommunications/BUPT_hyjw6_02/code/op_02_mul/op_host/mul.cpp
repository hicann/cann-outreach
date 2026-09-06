#include "register/op_def_registry.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
namespace {
constexpr uint32_t DATA_BLOCK_BYTES = 32;
constexpr uint32_t VECTOR_REPEAT_BYTES = 256;
constexpr uint32_t BLOCK_DIM = 8;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    const gert::Tensor *tensorX = context->GetRequiredInputTensor(0);
    if (tensorX == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType dtypeX = tensorX->GetDataType();
    const uint32_t dtypeSize = static_cast<uint32_t>(ge::GetSizeByDataType(dtypeX));
    if (dtypeSize == 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength = static_cast<uint32_t>(tensorX->GetShapeSize());
    const uint32_t alignNum = DATA_BLOCK_BYTES / dtypeSize;
    if (totalLength == 0 || totalLength % alignNum != 0) {
        return ge::GRAPH_FAILED;
    }

    if (totalLength % BLOCK_DIM != 0) {
        return ge::GRAPH_FAILED;
    }

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    tiling->blockLength = totalLength / BLOCK_DIM;
    if (tiling->blockLength % alignNum != 0 ||
        (tiling->blockLength * dtypeSize) % VECTOR_REPEAT_BYTES != 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    context->SetBlockDim(BLOCK_DIM);
    size_t *workspaceSizes = context->GetWorkspaceSizes(1);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    if (xShape == nullptr || zShape == nullptr) {
        return GRAPH_FAILED;
    }
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED).DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}  // namespace ops
