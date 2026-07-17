
#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static constexpr uint32_t TARGET_TILE_LENGTH = 8192;
static constexpr uint32_t MIN_TILE_LENGTH = 256;
static constexpr uint32_t MAX_TILE_NUM = 255;
static constexpr uint32_t MAX_BLOCK_DIM = 40;
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTemplateTilingData *tiling = context->GetTilingData<AddCustomTemplateTilingData>();
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    uint32_t blockDim = MAX_BLOCK_DIM;
    while (blockDim > 1 && totalLength / blockDim < TARGET_TILE_LENGTH) {
        blockDim >>= 1;
    }
    context->SetBlockDim(blockDim);

    uint32_t blockLength = totalLength / blockDim;

    uint32_t tileNum = (blockLength + TARGET_TILE_LENGTH - 1) / TARGET_TILE_LENGTH;
    if (tileNum < 1) {
        tileNum = 1;
    } else if (tileNum > MAX_TILE_NUM) {
        tileNum = MAX_TILE_NUM;
    }

    uint32_t tileLength = blockLength / tileNum;
    while (tileNum > 1 && tileLength < MIN_TILE_LENGTH) {
        tileNum >>= 1;
        tileLength = blockLength / tileNum;
    }

    tiling->totalLength = totalLength;
    tiling->tileNum = tileNum;

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
