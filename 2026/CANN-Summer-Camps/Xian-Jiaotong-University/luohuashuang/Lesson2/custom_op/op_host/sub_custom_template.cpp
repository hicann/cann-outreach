#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    
    // 使用 int64_t 累加，避免溢出
    int64_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++) {
        data_sz *= x1_shape->GetStorageShape().GetDim(i);
    }
    
    // 范围校验：确保不超过 uint32_t 最大值
    OP_TILING_CHECK(data_sz <= UINT32_MAX, "data_sz=%ld exceeds uint32_t max", data_sz);
    tiling->size = static_cast<uint32_t>(data_sz);
    
    context->SetBlockDim(8);
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

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
}

namespace ops {
class SubCustomTemplate : public OpDef {
public:
    explicit SubCustomTemplate(const char* name) : OpDef(name)
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
            .SetTiling(optiling::TilingFunc)
            .SetKernel("sub_custom_template_fp16")
            .SetKernel("sub_custom_template_fp32");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SubCustomTemplate);
}