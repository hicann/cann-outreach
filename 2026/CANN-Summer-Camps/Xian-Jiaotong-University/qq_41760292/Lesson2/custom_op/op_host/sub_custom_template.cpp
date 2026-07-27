#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

    static ge::graphStatus TilingFunc(gert::TilingContext* context)
    {
        auto* tilingData = context->GetTilingData<SubCustomTemplateTilingData>();

        const auto* inputShape = context->GetInputShape(0);
        const auto& storageShape = inputShape->GetStorageShape();

        uint32_t elementCount = 1;
        for (int32_t dim = 0; dim < storageShape.GetDimNum(); ++dim) {
            elementCount *= storageShape.GetDim(dim);
        }

        tilingData->size = elementCount;

        context->SetBlockDim(8);

        size_t* workspaceSize = context->GetWorkspaceSizes(1);
        workspaceSize[0] = 0;

        return ge::GRAPH_SUCCESS;
    }

}  // namespace optiling

namespace ge {

    static graphStatus InferShape(gert::InferShapeContext* context)
    {
        const auto* inputShape = context->GetInputShape(0);
        auto* outputShape = context->GetOutputShape(0);

        *outputShape = *inputShape;

        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext* context)
    {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return GRAPH_SUCCESS;
    }

}  // namespace ge

namespace ops {

    class SubCustomTemplate : public OpDef {
    public:
        explicit SubCustomTemplate(const char* name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });

            this->Input("y")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });

            this->Output("z")
                .ParamType(REQUIRED)
                .DataType({ ge::DT_FLOAT16, ge::DT_FLOAT })
                .Format({ ge::FORMAT_ND, ge::FORMAT_ND })
                .UnknownShapeFormat({ ge::FORMAT_ND, ge::FORMAT_ND });

            this->SetInferShape(ge::InferShape)
                .SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b")
                .AddConfig("ascend910_93");
        }
    };

    OP_ADD(SubCustomTemplate);

}  // namespace ops