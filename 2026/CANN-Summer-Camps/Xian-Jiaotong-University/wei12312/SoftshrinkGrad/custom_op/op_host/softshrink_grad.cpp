#include <algorithm>
#include <cstdint>
#include <limits>

#include "../op_kernel/softshrink_grad_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t MAX_BLOCK_DIM = 40;
constexpr uint32_t TILE_LENGTH = 2048;
constexpr uint32_t BLOCK_ALIGN_ELEMENTS = 16;
constexpr uint32_t TILES_PER_CORE_TARGET = 4;

uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *gradShape = context->GetInputShape(0);
    const gert::StorageShape *selfShape = context->GetInputShape(1);
    SoftshrinkGradTilingData *tiling = context->GetTilingData<SoftshrinkGradTilingData>();
    if (gradShape == nullptr || selfShape == nullptr || tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape &gradOriginShape = gradShape->GetOriginShape();
    const gert::Shape &selfOriginShape = selfShape->GetOriginShape();
    const int64_t gradLength = gradOriginShape.GetShapeSize();
    const int64_t selfLength = selfOriginShape.GetShapeSize();
    if (gradLength < 0 || gradLength != selfLength ||
        gradOriginShape.GetDimNum() != selfOriginShape.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < gradOriginShape.GetDimNum(); ++i) {
        if (gradOriginShape.GetDim(i) != selfOriginShape.GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }

    float lambd = 0.5F;
    if (context->GetAttrs() != nullptr) {
        const float *attr = context->GetAttrs()->GetAttrPointer<float>(0);
        if (attr != nullptr) {
            lambd = *attr;
        }
    }
    if (lambd < 0.0F) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t totalLength = static_cast<uint64_t>(gradLength);
    const uint64_t elementsPerCoreTarget =
        static_cast<uint64_t>(TILE_LENGTH) * TILES_PER_CORE_TARGET;
    uint32_t blockDim = 1;
    if (totalLength > 0) {
        const uint64_t requiredBlocks =
            (totalLength + elementsPerCoreTarget - 1) / elementsPerCoreTarget;
        blockDim = static_cast<uint32_t>(
            std::min<uint64_t>(MAX_BLOCK_DIM, std::max<uint64_t>(1, requiredBlocks)));
    }

    uint64_t blockLength = 0;
    if (totalLength > 0) {
        blockLength = AlignUp((totalLength + blockDim - 1) / blockDim,
                              BLOCK_ALIGN_ELEMENTS);
    }
    if (blockLength > std::numeric_limits<uint32_t>::max()) {
        return ge::GRAPH_FAILED;
    }

    context->SetBlockDim(blockDim);
    tiling->totalLength = totalLength;
    tiling->blockLength = static_cast<uint32_t>(blockLength);
    tiling->tileLength = TILE_LENGTH;
    tiling->lambd = lambd;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr ||
        context->GetInputShape(1) == nullptr || context->GetOutputShape(0) == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *gradShape = context->GetInputShape(0);
    const gert::Shape *selfShape = context->GetInputShape(1);
    if (gradShape->GetDimNum() != selfShape->GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < gradShape->GetDimNum(); ++i) {
        if (gradShape->GetDim(i) != selfShape->GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    *context->GetOutputShape(0) = *gradShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr || context->GetInputDataType(0) != context->GetInputDataType(1)) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SoftshrinkGrad : public OpDef {
public:
    explicit SoftshrinkGrad(const char *name) : OpDef(name)
    {
        this->Input("gradOutput")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("gradInput")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("lambd").AttrType(OPTIONAL).Float(0.5F);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(SoftshrinkGrad);
}  // namespace ops
