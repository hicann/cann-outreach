#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t BLOCK_SIZE_TARGET = 8192;
static constexpr uint32_t BLOCK_SIZE_MIN    = 256;
static constexpr uint32_t TILE_NUM_MAX      = 255;
static constexpr uint32_t CORE_NUM_MAX      = 40;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTemplateTilingData *tiling = context->GetTilingData<AddCustomTemplateTilingData>();
    uint32_t totalLen = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    uint32_t usedCores = CORE_NUM_MAX;
    while (usedCores > 1 && totalLen / usedCores < BLOCK_SIZE_TARGET) {
        usedCores >>= 1;
    }
    context->SetBlockDim(usedCores);

    uint32_t perCoreLen = totalLen / usedCores;
    uint32_t tileCount = (perCoreLen + BLOCK_SIZE_TARGET - 1) / BLOCK_SIZE_TARGET;
    if (tileCount < 1) tileCount = 1;
    else if (tileCount > TILE_NUM_MAX) tileCount = TILE_NUM_MAX;

    uint32_t tileSz = perCoreLen / tileCount;
    while (tileCount > 1 && tileSz < BLOCK_SIZE_MIN) {
        tileCount >>= 1;
        tileSz = perCoreLen / tileCount;
    }

    tiling->totalLength = totalLen;
    tiling->tileNum = tileCount;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* c) {
    *c->GetOutputShape(0) = *c->GetInputShape(0); return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* c) {
    c->SetOutputDataType(0, c->GetInputDataType(0)); return GRAPH_SUCCESS;
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
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(AddCustomTemplate);
}
