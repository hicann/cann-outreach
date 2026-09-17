#include "gcd123_tiling.h"

#include "register/op_def_registry.h"

namespace optiling {

constexpr uint32_t GCD123_NDIM = 4;
constexpr uint32_t GCD123_BLOCK_DIM = 8;

static bool LoadDims(const gert::Shape& shape, int64_t* dims)
{
    int64_t dimNum = shape.GetDimNum();
    if (dimNum <= 0 || dimNum > GCD123_NDIM) {
        return false;
    }

    for (uint32_t i = 0; i < GCD123_NDIM; ++i) {
        dims[i] = 1;
    }

    int64_t offset = GCD123_NDIM - dimNum;
    for (int64_t i = 0; i < dimNum; ++i) {
        dims[offset + i] = shape.GetDim(i);
    }
    return true;
}

static bool BroadcastDims(const int64_t* selfDims, const int64_t* otherDims, int64_t* outDims)
{
    for (uint32_t i = 0; i < GCD123_NDIM; ++i) {
        int64_t selfDim = selfDims[i];
        int64_t otherDim = otherDims[i];
        if (selfDim == otherDim || selfDim == 1 || otherDim == 1) {
            outDims[i] = selfDim > otherDim ? selfDim : otherDim;
        } else {
            return false;
        }
    }
    return true;
}

static int64_t Product(const int64_t* dims)
{
    int64_t length = 1;
    for (uint32_t i = 0; i < GCD123_NDIM; ++i) {
        length *= dims[i];
    }
    return length;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    Gcd123TilingData tiling;
    int64_t selfShape[GCD123_NDIM] = {1, 1, 1, 1};
    int64_t otherShape[GCD123_NDIM] = {1, 1, 1, 1};
    int64_t outShape[GCD123_NDIM] = {1, 1, 1, 1};

    if (!LoadDims(context->GetInputShape(0)->GetStorageShape(), selfShape) ||
        !LoadDims(context->GetInputShape(1)->GetStorageShape(), otherShape) ||
        !BroadcastDims(selfShape, otherShape, outShape)) {
        return ge::GRAPH_FAILED;
    }

    int64_t selfLength = Product(selfShape);
    int64_t otherLength = Product(otherShape);
    int64_t totalLength = Product(outShape);
    if (selfLength <= 0 || otherLength <= 0 || totalLength <= 0) {
        return ge::GRAPH_FAILED;
    }

    uint32_t blockDim = totalLength < GCD123_BLOCK_DIM ? static_cast<uint32_t>(totalLength) : GCD123_BLOCK_DIM;
    if (blockDim == 0) {
        blockDim = 1;
    }

    tiling.set_selfLength(selfLength);
    tiling.set_otherLength(otherLength);
    tiling.set_totalLength(totalLength);
    tiling.set_selfShape(selfShape);
    tiling.set_otherShape(otherShape);
    tiling.set_outShape(outShape);

    context->SetBlockDim(blockDim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context)
{
    int64_t selfDims[optiling::GCD123_NDIM] = {1, 1, 1, 1};
    int64_t otherDims[optiling::GCD123_NDIM] = {1, 1, 1, 1};
    int64_t outDims[optiling::GCD123_NDIM] = {1, 1, 1, 1};

    if (!optiling::LoadDims(*context->GetInputShape(0), selfDims) ||
        !optiling::LoadDims(*context->GetInputShape(1), otherDims) ||
        !optiling::BroadcastDims(selfDims, otherDims, outDims)) {
        return GRAPH_FAILED;
    }

    gert::Shape* outShape = context->GetOutputShape(0);
    outShape->SetDimNum(optiling::GCD123_NDIM);
    for (uint32_t i = 0; i < optiling::GCD123_NDIM; ++i) {
        outShape->SetDim(i, outDims[i]);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class gcd123 : public OpDef {
public:
    explicit gcd123(const char* name) : OpDef(name)
    {
        this->Input("self")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("other")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Output("out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910")
            .AddConfig("ascend310p")
            .AddConfig("ascend310b")
            .AddConfig("ascend910b");
    }
};

OP_ADD(gcd123);

}  // namespace ops
