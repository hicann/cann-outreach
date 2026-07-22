
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    // 1. 获取输入数据的 Shape 信息，并计算总元素个数 (totalLength)
    const gert::StorageShape* shape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= shape->GetStorageShape().GetDim(i);
    }

    // 2. 设置 AI Core 的工作数量为给定的 BLOCK_DIM (8)[cite: 4]
    context->SetBlockDim(BLOCK_DIM);

    // 3. 实例化普通结构体，直接进行变量赋值！[cite: 5]
    TanhCustomTilingData tiling;
    tiling.totalLength = totalLength;
    tiling.tileNum = TILE_NUM; // TILE_NUM 在该 cpp 顶部已定义为 8[cite: 4]

    // 4. 将结构体直接拷贝到 Context 的 RawTilingData 缓冲区中
    uint32_t tilingSize = sizeof(TanhCustomTilingData);
    
    // 确保目标缓冲区容量足够，避免内存越界
    if (context->GetRawTilingData()->GetCapacity() < tilingSize) {
        return ge::GRAPH_FAILED;
    }
    
    // 使用 std::memcpy 进行内存拷贝，直接复制结构体数据
    std::memcpy(context->GetRawTilingData()->GetData(), &tiling, tilingSize);
    context->GetRawTilingData()->SetDataSize(tilingSize);
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
