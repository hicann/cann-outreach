#include <cstdint>
#include <limits>

#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
// ✅ 优化配置
constexpr uint32_t BLOCK_DIM = 40;      // 充分利用 40 个 AI Core
constexpr uint32_t TILE_NUM = 4;        // 增加 tile 数量，减少循环开销
constexpr uint32_t BUFFER_NUM = 2;      // 双缓冲

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* inputShape = context->GetInputShape(0);
    AddCustomTemplateTilingData* tiling = context->GetTilingData<AddCustomTemplateTilingData>();
    if (inputShape == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t totalLength = inputShape->GetOriginShape().GetShapeSize();
    
    // ✅ 数据校验
    if (totalLength <= 0 ||
        static_cast<uint64_t>(totalLength) > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    // ✅ 设置多核并行
    context->SetBlockDim(BLOCK_DIM);
    
    // ✅ 设置 Tiling 参数
    tiling->totalLength = static_cast<uint32_t>(totalLength);
    
    // ✅ 关键优化：调整 tileNum 使数据切分更精细
    // 总元素: 45 * 20480 = 921,600
    // 每核: 921,600 / 40 = 23,040
    // tileNum = 4 → 每 tile 5,760 元素 (23,040 字节)
    // 这样每个 tile 足够大，能充分利用带宽，又不会溢出 UB
    tiling->tileNum = TILE_NUM;
    
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
class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AddCustomTemplate);
}