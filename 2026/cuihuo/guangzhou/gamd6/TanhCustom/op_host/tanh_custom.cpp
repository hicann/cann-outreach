
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 1. 获取输入总元素个数，作为多核切分的总数据量（totalLength）
    const gert::Shape& inputShape = context->GetInputShape(0)->GetOriginShape();
    uint32_t totalLength = static_cast<uint32_t>(inputShape.GetShapeSize());
    if (totalLength == 0) {
        return ge::GRAPH_FAILED;
    }

    // 2. 填充 Tiling 结构体：本算子使用固定核数 BLOCK_DIM，每核数据再按 TILE_NUM 个超块切分
    TanhCustomTilingData tilingData;
    tilingData.totalLength = totalLength;
    tilingData.tileNum = TILE_NUM;

    // 3. 将 Tiling 数据序列化到 context 的 raw tiling buffer 中（kernel 侧 GET_TILING_DATA 读取）
    auto rawTiling = context->GetRawTilingData();
    if (rawTiling->GetCapacity() < sizeof(TanhCustomTilingData)) {
        return ge::GRAPH_FAILED;
    }
    TanhCustomTilingData* tilingPtr =
        reinterpret_cast<TanhCustomTilingData*>(rawTiling->GetData());
    *tilingPtr = tilingData;
    rawTiling->SetDataSize(sizeof(TanhCustomTilingData));

    // 4. 设置参与计算的核数（blockDim）
    context->SetBlockDim(BLOCK_DIM);

    // 5. 本算子无需额外的 workspace，置 0 即可
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;
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
