
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 获取 TilingData 结构体指针
    auto tiling = context->GetTilingData<TanhCustomTilingData>();
    // 获取输入张量的总元素个数
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();
    // 设置核函数启动的 AI Core 数量（8 核）
    context->SetBlockDim(BLOCK_DIM);

    // 填充 Tiling 参数
    tiling->totalElements = totalLength;
    tiling->blockDim      = BLOCK_DIM;
    tiling->tileNum       = TILE_NUM;
    // 可选：计算每个 core 处理的元素数（向上取整）
    tiling->blockSize = (totalLength + BLOCK_DIM - 1) / BLOCK_DIM;
    // 可选：计算每个 tile 处理的元素数（向上取整）
    tiling->tileSize  = (tiling->blockSize + TILE_NUM - 1) / TILE_NUM;

    // 设置 Tiling Key（用于区分不同的切分策略，此处固定为 1）
    context->SetTilingKey(1);
    // 获取 Workspace 大小（本例不需要额外workspace，设为0）
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
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

