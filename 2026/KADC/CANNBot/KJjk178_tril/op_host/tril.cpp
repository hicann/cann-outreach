#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/tril_tiling.h"

#include "../op_kernel/tiling_key_tril.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        uint32_t length_x = tensor_x->GetShapeSize();

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        int32_t diagonal = 0;
        if (attrs->GetInt(0) != nullptr) {
            diagonal = static_cast<int32_t>(*attrs->GetInt(0));
        }

        const gert::StorageShape *x_shape = context->GetInputShape(0);
        auto shape_dims = x_shape->GetStorageShape();
        int32_t rank = shape_dims.GetDimNum();
        uint32_t M = shape_dims.GetDim(rank - 2);
        uint32_t N = shape_dims.GetDim(rank - 1);
        uint32_t batchSize = 1;
        for (int32_t i = 0; i < rank - 2; i++) {
            batchSize *= shape_dims.GetDim(i);
        }

        uint32_t totalRows = batchSize * M;
        uint32_t blockDim = std::min((uint32_t)num_cores_aiv, totalRows);
        if (blockDim == 0) blockDim = 1;
        uint32_t rowsPerCore = totalRows / blockDim;
        uint32_t tailRows = totalRows - rowsPerCore * (blockDim - 1);

        uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        ASCENDC_TPL_SEL_PARAM(context, DT_X);

        TrilTilingData *tiling = context->GetTilingData<TrilTilingData>();
        tiling->totalLength = length_x;
        tiling->M = M;
        tiling->N = N;
        tiling->batchSize = batchSize;
        tiling->diagonal = diagonal;
        tiling->totalRows = totalRows;
        tiling->rowsPerCore = rowsPerCore;
        tiling->tailRows = tailRows;

        context->SetBlockDim(blockDim);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Tril : public OpDef {
    public:
        explicit Tril(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("diagonal").AttrType(OPTIONAL).Int();
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Tril);
}  // namespace ops
