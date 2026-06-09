#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstring>

namespace optiling {
constexpr int64_t UB_BUFFER_NUM = 7;
constexpr int64_t TYPE_SIZE = 2;
constexpr int64_t BLOCK_ALIGN_NUM = 16;
constexpr uint32_t DEFAULT_BLOCK_DIM = 8;

static int64_t CeilDiv(int64_t value, int64_t factor)
{
    return (value + factor - 1) / factor;
}

static int64_t FloorAlign(int64_t value, int64_t factor)
{
    return value / factor * factor;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int64_t totalNum = inputShape->GetStorageShape().GetShapeSize();
    if (totalNum <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint64_t ubSize = 0;
    int64_t coreNum = DEFAULT_BLOCK_DIM;
    auto platformInfo = context->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        int64_t platformCoreNum = ascendcPlatform.GetCoreNumAiv();
        if (platformCoreNum > 0) {
            coreNum = platformCoreNum;
        }
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    }

    if (ubSize == 0) {
        ubSize = 128 * 1024;
    }

    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    int64_t usedCoreNum = std::min<int64_t>(coreNum, CeilDiv(totalNum, blockFactor));
    int64_t ubFactor = FloorAlign(static_cast<int64_t>(ubSize) / TYPE_SIZE / UB_BUFFER_NUM, BLOCK_ALIGN_NUM);
    if (ubFactor <= 0) {
        ubFactor = BLOCK_ALIGN_NUM;
    }

    TanhCustomTilingData* tiling = context->GetTilingData<TanhCustomTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    size_t* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::memset(tiling, 0, sizeof(TanhCustomTilingData));
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    workspaceSizes[0] = 0;
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* yShape = context->GetOutputShape(0);
    *yShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
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
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND})
            .AutoContiguous();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(TanhCustom);
}
