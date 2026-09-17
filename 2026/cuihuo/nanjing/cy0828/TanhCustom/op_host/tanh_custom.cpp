
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 1. 入参空指针校验
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 2. 输入/输出tensor合法性校验
    const gert::Tensor* inputTensor = context->GetInputTensor(0);
    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    if (inputTensor == nullptr || outputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t inputSize = inputTensor->GetShapeSize();
    int64_t outputSize = outputShape->GetStorageShape().GetShapeSize();
    if (inputSize <= 0 || inputSize != outputSize) {
        return ge::GRAPH_FAILED;
    }
    // 3. 纯Vector算子无需GM workspace，正确赋值workspaceSize出参
    size_t* workspaceSize = context->GetWorkspaceSizes(0);
    if (workspaceSize == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *workspaceSize = 0;
    // 4. 设置SIMD核数(blockDim)，否则kernel launch报blockDim=0非法
    ge::graphStatus status = context->SetSimdNumBlocks(BLOCK_DIM);
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }
    // 5. 填充tiling数据，kernel侧经GET_TILING_DATA读取
    TanhCustomTilingData* tilingData = context->GetTilingData<TanhCustomTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tilingData->blockDim = BLOCK_DIM;
    tilingData->totalSize = (uint64_t)inputSize;
    return ge::GRAPH_SUCCESS;
}
}


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
