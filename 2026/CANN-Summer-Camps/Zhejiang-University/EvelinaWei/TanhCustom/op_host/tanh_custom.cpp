
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
namespace {
constexpr int64_t BLOCK_DIM = 8;
constexpr int64_t TILE_NUM = 8;

int64_t CeilDiv(int64_t value, int64_t factor)
{
    return (value + factor - 1) / factor;
}
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::Tensor* inputTensor = context->GetInputTensor(0);
    if (inputTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t totalNum = inputTensor->GetShapeSize();
    TanhCustomTilingData* tiling = context->GetTilingData<TanhCustomTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t blockFactor = CeilDiv(totalNum, BLOCK_DIM);
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = CeilDiv(blockFactor, TILE_NUM);

    size_t workspaceNum = context->GetWorkspaceNum();
    if (workspaceNum > 0) {
        size_t* curWorkspace = context->GetWorkspaceSizes(workspaceNum);
        if (curWorkspace != nullptr) {
            curWorkspace[0] = 0;
        }
    }

    context->SetBlockDim(BLOCK_DIM);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class TanhCustom : public OpDef {
public:
    explicit TanhCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(TanhCustom);
}
