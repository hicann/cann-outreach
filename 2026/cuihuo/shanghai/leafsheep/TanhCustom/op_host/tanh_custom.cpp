
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstring>


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
const uint32_t BUFFER_NUM = 2;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 1. 获取输入张量的总元素数
    auto inputTensor = context->GetInputTensor(0);
    uint32_t totalLength = static_cast<uint32_t>(inputTensor->GetShapeSize());

    // 2. 计算分块参数
    uint32_t blockLength = totalLength / BLOCK_DIM;
    uint32_t tileLength  = blockLength / TILE_NUM / BUFFER_NUM;

    // 3. 填充 TilingData 结构体
    TanhCustomTilingData tilingData;
    tilingData.totalLength = totalLength;
    tilingData.tileNum     = TILE_NUM;
    tilingData.tileLength  = tileLength;

    // 4. 写回 context
    context->SetBlockDim(BLOCK_DIM);
    auto tilingDataPtr = context->GetRawTilingData();
    tilingDataPtr->SetDataSize(sizeof(TanhCustomTilingData));
    errno_t ret = memcpy_s(tilingDataPtr->GetData(),
                           tilingDataPtr->GetCapacity(),
                           &tilingData,
                           sizeof(TanhCustomTilingData));
    if (ret != EOK) {
        return ge::GRAPH_FAILED;
    }

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

