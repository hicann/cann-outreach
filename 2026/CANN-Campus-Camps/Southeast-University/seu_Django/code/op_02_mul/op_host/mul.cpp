#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const auto* xShape = context->GetInputShape(0);
    const auto* yShape = context->GetInputShape(1);
    const auto* xDesc = context->GetInputDesc(0);
    const auto* yDesc = context->GetInputDesc(1);
    if (!xShape || !yShape || !xDesc || !yDesc ||
        !context->GetPlatformInfo()) {
        return ge::GRAPH_FAILED;
    }
    const auto& xs = xShape->GetStorageShape();
    const auto& ys = yShape->GetStorageShape();
    if (xs.GetDimNum() != ys.GetDimNum()) {
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0; i < xs.GetDimNum(); ++i) {
        if (xs.GetDim(i) < 0 || xs.GetDim(i) != ys.GetDim(i)) {
            return ge::GRAPH_FAILED;
        }
    }
    const auto dtype = xDesc->GetDataType();
    if (dtype != yDesc->GetDataType() ||
        (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t size = xs.GetShapeSize();
    if (size < 0 || static_cast<uint64_t>(size) > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t availableCores = platform.GetCoreNumAiv();
    if (availableCores == 0) {
        return ge::GRAPH_FAILED;
    }
    // 三个双缓冲队列，每个 buffer 为 8 KiB，共使用 48 KiB UB。
    const uint32_t elementBytes = dtype == ge::DT_FLOAT ? 4 : 2;
    const uint32_t tileLength = 8192 / elementBytes;
    const uint32_t tileCount = static_cast<uint32_t>(
        (static_cast<uint64_t>(size) + tileLength - 1) / tileLength);
    const uint32_t coreNum = tileCount == 0 ? 1 :
        (tileCount < availableCores ? tileCount : availableCores);

    auto* tiling = context->GetTilingData<MulTilingData>();
    if (!tiling || !context->GetRawTilingData()) {
        return ge::GRAPH_FAILED;
    }
    tiling->totalLength = static_cast<uint32_t>(size);
    tiling->tileNum = tileCount;
    tiling->tileLength = tileLength;
    tiling->coreNum = coreNum;
    context->GetRawTilingData()->SetDataSize(sizeof(MulTilingData));
    context->SetBlockDim(coreNum);
    const uint32_t DT_X = static_cast<uint32_t>(dtype);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);
    auto* workspace = context->GetWorkspaceSizes(1);
    if (!workspace) {
        return ge::GRAPH_FAILED;
    }
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context)
{
    const auto* xShape = context->GetInputShape(0);
    auto* zShape = context->GetOutputShape(0);
    if (!xShape || !zShape) {
        return GRAPH_FAILED;
    }
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto dtype = context->GetInputDataType(0);
    if (dtype != context->GetInputDataType(1) ||
        (dtype != DT_FLOAT && dtype != DT_FLOAT16)) {
        return GRAPH_FAILED;
    }
    context->SetOutputDataType(0, dtype);
    return GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char* name) : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z").ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
} // namespace ops
