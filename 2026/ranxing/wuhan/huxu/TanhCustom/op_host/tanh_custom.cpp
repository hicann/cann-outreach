#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include <cstring>   // for memcpy

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 1. 获取输入张量总元素数
    uint32_t totalLength = context->GetInputTensor(0)->GetShapeSize();

    // 2. 设置并行核心数
    context->SetBlockDim(BLOCK_DIM);

    // 3. 填充自定义 Tiling 数据结构
    TanhCustomTilingData tiling;
    tiling.totalLength = totalLength;
    tiling.tileNum = TILE_NUM;

    // 4. 将数据写入 RawTilingData 缓冲区
    auto* rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    rawTiling->SetDataSize(sizeof(TanhCustomTilingData));
    void* data = rawTiling->GetData();    // GetData() 返回 void*
    if (data == nullptr) {
        return ge::GRAPH_FAILED;
    }
    memcpy(data, &tiling, sizeof(tiling)); // memcpy 接受 void*

    // 5. 设置 Workspace 大小（无需额外内存）
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
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
