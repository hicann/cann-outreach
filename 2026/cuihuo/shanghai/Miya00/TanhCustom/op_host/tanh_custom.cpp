#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 1. 取输入 shape（CANN 9.0：GetInputShape 返回 StorageShape*）
    const gert::StorageShape* storageShape = context->GetInputShape(0);
    const gert::Shape& shape = storageShape->GetShape();

    // 2. 算总元素个数
    uint32_t totalLength = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalLength *= shape.GetDim(i);
    }

    // 3. 把 tiling 数据写入 context（CANN 9.0 正确方式）
    TanhCustomTilingData* tiling =
        reinterpret_cast<TanhCustomTilingData*>(context->GetRawTilingData()->GetData());
    tiling->totalLength = totalLength;
    tiling->tileNum = TILE_NUM;
    context->GetRawTilingData()->SetDataSize(sizeof(TanhCustomTilingData));

    // 4. 设置 block 维数
    context->SetBlockDim(BLOCK_DIM);

    return ge::GRAPH_SUCCESS;
}
}

// ===== 下面 InferShape / InferDataType / OpDef 保持你原文件不动 =====
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