#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

// Optimal tile size for L1 cache utilization
// Ascend 910B provides approximately 192KB L1 per AI core.
// Using double buffering with 3 queues:
//   fp16 (2 bytes): 2 * 3 * 2 * tileLen = 12 * tileLen → 8K = 96KB (good fit)
//   fp32 (4 bytes): 2 * 3 * 4 * tileLen = 24 * tileLen → 8K = 192KB (tight but acceptable)
static constexpr uint32_t OPTIMAL_TILE_SIZE = 8192;
static constexpr uint32_t MIN_TILE_UNIT = 256;
static constexpr uint32_t MAX_TILE_COUNT = 255;
static constexpr uint32_t MAX_BLOCK_DIM = 40;

static ge::graphStatus CalculateTilingStrategy(gert::TilingContext* ctx)
{
    AddCustomTemplateTilingData *tilingConfig = ctx->GetTilingData<AddCustomTemplateTilingData>();
    uint32_t totalElements = ctx->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // Dynamically adjust block dimension based on input size
    uint32_t blockDim = MAX_BLOCK_DIM;
    while (blockDim > 1 && totalElements / blockDim < OPTIMAL_TILE_SIZE) {
        blockDim >>= 1;
    }
    ctx->SetBlockDim(blockDim);

    uint32_t blockSize = totalElements / blockDim;

    // Calculate tile count to achieve target tile size
    uint32_t tileCount = (blockSize + OPTIMAL_TILE_SIZE - 1) / OPTIMAL_TILE_SIZE;
    if (tileCount < 1) {
        tileCount = 1;
    } else if (tileCount > MAX_TILE_COUNT) {
        // Limit tile count to reduce scheduling overhead
        tileCount = MAX_TILE_COUNT;
    }

    // Ensure minimum tile size
    uint32_t tileSize = blockSize / tileCount;
    while (tileCount > 1 && tileSize < MIN_TILE_UNIT) {
        tileCount >>= 1;
        tileSize = blockSize / tileCount;
    }

    tilingConfig->totalLength = totalElements;
    tilingConfig->tileNum = tileCount;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferOutputShape(gert::InferShapeContext* ctx)
{
    const gert::Shape* inputShape = ctx->GetInputShape(0);
    gert::Shape* outputShape = ctx->GetOutputShape(0);
    
    *outputShape = *inputShape;
    
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferOutputDataType(gert::InferDataTypeContext *ctx)
{
    const auto inputDtype = ctx->GetInputDataType(0);
    ctx->SetOutputDataType(0, inputDtype);
    
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char* opName) : OpDef(opName)
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
            
        this->SetInferShape(ge::InferOutputShape)
            .SetInferDataType(ge::InferOutputDataType);
            
        this->AICore()
            .SetTiling(optiling::CalculateTilingStrategy);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AddCustomTemplate);

}  // namespace ops
