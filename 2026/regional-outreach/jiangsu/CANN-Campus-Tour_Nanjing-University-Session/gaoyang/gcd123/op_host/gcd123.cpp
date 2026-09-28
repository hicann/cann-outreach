/**
 * @file gcd123.cpp
 *
 * Host-side definition of the Gcd123 operator:
 *   - operator prototype registration (OpDef)
 *   - output shape inference (broadcast of two 4D ND tensors)
 *   - output data-type inference
 *   - Tiling implementation
 *
 * Semantics:
 *   out[i] = gcd(round|self[i]|, round|other[i]|)
 *   where self and other are broadcast element-wise along the
 *   4-dimensional ND shape [N4, N3, N2, N1].
 */
#include <cstdint>

#include "register/op_def_registry.h"

#include "gcd123_tiling.h"

namespace optiling {

// Logical number of AI cores to run the kernel on. The kernel side splits
// the flattened output linearly by (GetBlockNum, GetBlockIdx), so any value
// in [1, 65535] is valid. Adjust to the physical core count of your platform
// (GetCoreNumAiv) for best utilization.
const uint32_t BLOCK_DIM = 20;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    Gcd123TilingData tiling;

    const gert::Shape *selfShape = context->GetInputShape(0);
    const gert::Shape *otherShape = context->GetInputShape(1);
    const gert::Shape *outShape = context->GetOutputShape(0);

    const uint64_t totalLength = outShape->GetShapeSize();
    context->SetBlockDim(BLOCK_DIM);

    tiling.set_totalLength(totalLength);

    tiling.set_selfN4(static_cast<uint32_t>(selfShape->GetDim(0)));
    tiling.set_selfN3(static_cast<uint32_t>(selfShape->GetDim(1)));
    tiling.set_selfN2(static_cast<uint32_t>(selfShape->GetDim(2)));
    tiling.set_selfN1(static_cast<uint32_t>(selfShape->GetDim(3)));
    tiling.set_otherN4(static_cast<uint32_t>(otherShape->GetDim(0)));
    tiling.set_otherN3(static_cast<uint32_t>(otherShape->GetDim(1)));
    tiling.set_otherN2(static_cast<uint32_t>(otherShape->GetDim(2)));
    tiling.set_otherN1(static_cast<uint32_t>(otherShape->GetDim(3)));
    tiling.set_outN4(static_cast<uint32_t>(outShape->GetDim(0)));
    tiling.set_outN3(static_cast<uint32_t>(outShape->GetDim(1)));
    tiling.set_outN2(static_cast<uint32_t>(outShape->GetDim(2)));
    tiling.set_outN1(static_cast<uint32_t>(outShape->GetDim(3)));

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // No workspace is required by this operator.
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

// Broadcast the two 4D input shapes into the output shape.
// Returns GRAPH_FAILED when the shapes are not mutually broadcastable.
static ge::graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *selfShape = context->GetInputShape(0);
    const gert::Shape *otherShape = context->GetInputShape(1);
    gert::Shape *outShape = context->GetOutputShape(0);

    const int32_t numDims = 4;
    int64_t outDims[numDims] = {0, 0, 0, 0};

    for (int32_t d = 0; d < numDims; ++d) {
        const int64_t sd = selfShape->GetDim(d);
        const int64_t od = otherShape->GetDim(d);
        if (sd == od) {
            outDims[d] = sd;
        } else if (sd == 1) {
            outDims[d] = od;
        } else if (od == 1) {
            outDims[d] = sd;
        } else {
            return ge::GRAPH_FAILED;
        }
    }

    for (int32_t d = 0; d < numDims; ++d) {
        outShape->SetDim(d, outDims[d]);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

class Gcd123 : public OpDef {
public:
    explicit Gcd123(const char *name) : OpDef(name)
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
            .AddConfig("ascend910b")
            .AddConfig("ascend310p")
            .AddConfig("ascend310b");
    }
};

OP_ADD(Gcd123);

} // namespace ops
