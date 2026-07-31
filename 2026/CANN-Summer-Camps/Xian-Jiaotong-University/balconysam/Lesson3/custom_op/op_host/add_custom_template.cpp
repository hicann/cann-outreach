#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

// L1-friendly tile length target per element type.
// Ascend 910B has ~192KB L1 per AI core; with BUFFER_NUM=2 and 3 queues,
//   fp16 (2B): 2 * 3 * 2 * tileLength = 12 * tileLength → 8K ~= 96KB  (fits well)
//   fp32 (4B): 2 * 3 * 4 * tileLength = 24 * tileLength → 8K ~= 192KB (tight but ok)
static constexpr uint32_t TARGET_TILE_LENGTH = 8192;
static constexpr uint32_t MIN_TILE_LENGTH = 256;
static constexpr uint32_t MAX_TILE_NUM = 255;
static constexpr uint32_t MAX_BLOCK_DIM = 40;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTemplateTilingData *tiling = context->GetTilingData<AddCustomTemplateTilingData>();
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // Adaptive blockDim: scale down for small tensors so each block has enough work
    uint32_t blockDim = MAX_BLOCK_DIM;
    while (blockDim > 1 && totalLength / blockDim < TARGET_TILE_LENGTH) {
        blockDim >>= 1;  // halve the block count
    }
    context->SetBlockDim(blockDim);

    uint32_t blockLength = totalLength / blockDim;

    // Compute tileNum so each tile is roughly TARGET_TILE_LENGTH elements
    uint32_t tileNum = (blockLength + TARGET_TILE_LENGTH - 1) / TARGET_TILE_LENGTH;
    if (tileNum < 1) {
        tileNum = 1;
    } else if (tileNum > MAX_TILE_NUM) {
        // Cap to avoid scheduling overhead from too many tiny tiles;
        // this also bounds the tile length from below to MIN_TILE_LENGTH
        tileNum = MAX_TILE_NUM;
    }

    // Ensure minimum tile length
    uint32_t tileLength = blockLength / tileNum;
    while (tileNum > 1 && tileLength < MIN_TILE_LENGTH) {
        tileNum >>= 1;
        tileLength = blockLength / tileNum;
    }

    tiling->totalLength = totalLength;
    tiling->tileNum = tileNum;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

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

}  // namespace ge

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

}  // namespace ops
