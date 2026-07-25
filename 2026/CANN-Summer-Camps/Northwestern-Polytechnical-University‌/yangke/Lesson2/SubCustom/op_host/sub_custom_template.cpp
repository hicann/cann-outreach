#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/sub_custom_template_tiling.h"
#include <algorithm>
#include <cstdio>

namespace optiling {
static constexpr uint32_t GM_ALIGN_SIZE = 512;
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t QUEUE_NUM = 3;
static constexpr uint32_t FLOAT32_SIZE = 4;
static constexpr uint32_t BLOCK_SIZE = 32;
static constexpr uint32_t EXPECT_DIM_NUM = 2;
static constexpr int64_t EXPECT_DIM0 = 8;
static constexpr int64_t EXPECT_DIM1 = 2048;
static constexpr uint32_t UB_RESERVED_BYTES = 0;
static constexpr uint32_t TILING_KEY_VECTOR = 1;

static void LogError(const char *msg)
{
    std::fprintf(stderr, "[SubCustomTemplate][Host][Error] %s\n", msg);
    std::fflush(stderr);
}

static bool IsExpectedShape(const gert::Shape &shape)
{
    return shape.GetDimNum() == EXPECT_DIM_NUM &&
           shape.GetDim(0) == EXPECT_DIM0 &&
           shape.GetDim(1) == EXPECT_DIM1;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    if (context == nullptr) {
        LogError("context is null");
        return ge::GRAPH_FAILED;
    }
    if (context->GetInputShape(0) == nullptr || context->GetInputShape(1) == nullptr ||
        context->GetInputTensor(0) == nullptr || context->GetInputTensor(1) == nullptr) {
        LogError("input shape or tensor is null");
        return ge::GRAPH_FAILED;
    }

    const auto xShape = context->GetInputShape(0)->GetStorageShape();
    const auto yShape = context->GetInputShape(1)->GetStorageShape();
    if (!IsExpectedShape(xShape) || !IsExpectedShape(yShape)) {
        LogError("only shape (8, 2048) is supported");
        return ge::GRAPH_FAILED;
    }

    ge::DataType xType = context->GetInputTensor(0)->GetDataType();
    ge::DataType yType = context->GetInputTensor(1)->GetDataType();
    if (xType != yType || (xType != ge::DT_FLOAT16 && xType != ge::DT_FLOAT)) {
        LogError("only same dtype float16/float32 inputs are supported");
        return ge::GRAPH_FAILED;
    }

    uint64_t totalLength = xShape.GetShapeSize();
    uint32_t dataTypeSize = ge::GetSizeByDataType(xType);
    if (totalLength == 0 || dataTypeSize == 0) {
        LogError("invalid total length or dtype size");
        return ge::GRAPH_FAILED;
    }

    uint32_t alignedNum = BLOCK_SIZE / dataTypeSize;
    uint32_t alignGm = GM_ALIGN_SIZE / dataTypeSize;

    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendPlatform.GetCoreNumAiv();
    uint64_t ubNum = 0;
    ascendPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubNum);
    if (coreNum == 0 || ubNum == 0) {
        LogError("invalid platform core num or ub size");
        return ge::GRAPH_FAILED;
    }

    uint64_t effectiveUbNum = ubNum > UB_RESERVED_BYTES ? (ubNum - UB_RESERVED_BYTES) : ubNum;
    uint64_t ubPerElement = BUFFER_NUM * QUEUE_NUM * static_cast<uint64_t>(dataTypeSize);
    if (xType == ge::DT_FLOAT16) {
        ubPerElement += 3ULL * FLOAT32_SIZE;
    }

    uint32_t tileLength = static_cast<uint32_t>(effectiveUbNum / ubPerElement);
    tileLength = (tileLength / alignedNum) * alignedNum;
    if (tileLength == 0) {
        LogError("tileLength is zero");
        return ge::GRAPH_FAILED;
    }

    uint32_t needCoreNum = static_cast<uint32_t>((totalLength + tileLength - 1) / tileLength);
    needCoreNum = std::min(needCoreNum, coreNum);
    if (needCoreNum == 0) {
        LogError("needCoreNum is zero");
        return ge::GRAPH_FAILED;
    }

    uint64_t formerLength = totalLength / needCoreNum;
    formerLength = (formerLength + alignGm - 1) / alignGm * alignGm;
    if (formerLength == 0) {
        LogError("formerLength is zero");
        return ge::GRAPH_FAILED;
    }

    needCoreNum = static_cast<uint32_t>((totalLength + formerLength - 1) / formerLength);
    uint64_t tailLength = totalLength - static_cast<uint64_t>(needCoreNum - 1) * formerLength;

    if (formerLength < tileLength) {
        tileLength = static_cast<uint32_t>((formerLength + alignedNum - 1) / alignedNum * alignedNum);
    }

    uint32_t formerTileNum = static_cast<uint32_t>((formerLength + tileLength - 1) / tileLength);
    uint32_t formerLastTileLength = static_cast<uint32_t>(formerLength - static_cast<uint64_t>(formerTileNum - 1) * tileLength);
    uint32_t tailTileNum = static_cast<uint32_t>((tailLength + tileLength - 1) / tileLength);
    uint32_t tailLastTileLength = static_cast<uint32_t>(tailLength - static_cast<uint64_t>(tailTileNum - 1) * tileLength);

    SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();
    if (tiling == nullptr) {
        LogError("tiling ptr is null");
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = totalLength;
    tiling->formerLength = formerLength;
    tiling->tailLength = tailLength;
    tiling->tileLength = tileLength;
    tiling->formerTileNum = formerTileNum;
    tiling->tailTileNum = tailTileNum;
    tiling->formerLastTileLength = formerLastTileLength;
    tiling->tailLastTileLength = tailLastTileLength;

    context->SetBlockDim(needCoreNum);
    context->SetTilingKey(TILING_KEY_VECTOR);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    if (context == nullptr || context->GetInputShape(0) == nullptr || context->GetOutputShape(0) == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class SubCustomTemplate : public OpDef {
public:
    explicit SubCustomTemplate(const char *name) : OpDef(name)
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
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SubCustomTemplate);
}  // namespace ops
