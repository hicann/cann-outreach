#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    // 【核心修正点】：CANN 9.0.0 推荐直接使用 GetOriginShape().GetShapeSize() 
    // 一次性获取总元素个数，无需手动循环相乘维度，彻底避免类型和方法不匹配的问题。
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // 制定分片策略 (Tiling Strategy)
    uint32_t tileNum = 1;
    uint32_t blockLength = totalLength;

    // 如果数据量较大，则启用分片（以 8192 个元素为阈值，防止 UB 内存溢出）
    if (totalLength > 8192) {
        tileNum = TILE_NUM;
        // 向上取整计算每个分片的长度
        blockLength = (totalLength + tileNum - 1) / tileNum;
    }
    
    // 内存对齐：Ascend C 的 Vector 指令通常要求 32 字节对齐
    // 对于 float16 (2字节/元素)，32字节 = 16 个元素。因此需要向上对齐到 16 的倍数
    blockLength = (blockLength + 15) / 16 * 16;

    // 获取框架分配好的 Tiling 数据结构指针，并填充我们的“分片计划”
    TanhCustomTilingData* tilingData = context->GetTilingData<TanhCustomTilingData>();
    if (tilingData == nullptr) {
        return ge::GRAPH_FAILED;
    }
    
    tilingData->totalLength = totalLength;
    tilingData->tileNum = tileNum;
    tilingData->blockLength = blockLength;

    // 设置参与计算的 Block (AI Core) 数量
    context->SetBlockDim(BLOCK_DIM);

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