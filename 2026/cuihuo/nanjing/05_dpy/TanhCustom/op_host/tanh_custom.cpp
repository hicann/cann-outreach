
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

    // TODO: 考生自行补齐Tiling函数
    // 1. 获取输入数据量大小 (使用 GetShapeSize 获取总元素个数)
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    
    // 2. 设置 AI Core 的核心数 (Block Dim)
    context->SetBlockDim(8);

    // 3. 设置 Workspace (部分 CANN 版本强校验，没有用到也要设为0)
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
    }

    // 4. 实例化原生 Tiling 结构体并赋值
    TanhCustomTilingData tiling;
    tiling.totalLength = totalLength;
    tiling.tileNum = 8; // 注意：需要与 Kernel 侧初始化的切分份数一致

    // 5. 序列化 Tiling 数据：将数据拷贝到底层 context 缓冲区
    auto rawTilingData = context->GetRawTilingData();
    rawTilingData->SetDataSize(sizeof(TanhCustomTilingData));
    memcpy(rawTilingData->GetData(), &tiling, sizeof(TanhCustomTilingData));

    // 6. 【致命关键】必须告诉底层框架 Tiling 成功了！
    return ge::GRAPH_SUCCESS;
    // return ge::GRAPH_FAILED;
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
