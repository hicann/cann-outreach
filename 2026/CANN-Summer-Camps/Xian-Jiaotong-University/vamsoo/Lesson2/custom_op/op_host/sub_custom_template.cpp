#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"



namespace optiling {


static ge::graphStatus TilingFunc(
    gert::TilingContext* context)
{


    auto tiling =
        context->GetTilingData<SubCustomTemplateTilingData>();


    const gert::StorageShape* xShape =
        context->GetInputShape(0);



    int32_t size = 1;


    for(int i = 0;
        i < xShape->GetStorageShape().GetDimNum();
        i++)
    {
        size *=
            xShape->GetStorageShape().GetDim(i);
    }



    tiling->size = size;



    // 8个AI Core
    context->SetBlockDim(8);



    size_t *workspace =
        context->GetWorkspaceSizes(1);


    workspace[0] = 0;



    return ge::GRAPH_SUCCESS;

}

}



namespace ge {


static ge::graphStatus InferShape(
    gert::InferShapeContext* context)
{


    const gert::Shape* xShape =
        context->GetInputShape(0);



    gert::Shape* zShape =
        context->GetOutputShape(0);



    *zShape = *xShape;



    return GRAPH_SUCCESS;

}




static ge::graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{


    auto dtype =
        context->GetInputDataType(0);



    context->SetOutputDataType(
        0,
        dtype);



    return GRAPH_SUCCESS;

}


}





namespace ops {


class SubCustomTemplate : public OpDef
{


public:


    explicit SubCustomTemplate(
        const char* name)
        : OpDef(name)
    {



        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });



        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });



        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });



        this->SetInferShape(
            ge::InferShape)
            .SetInferDataType(
            ge::InferDataType);



        this->AICore()
            .SetTiling(
                optiling::TilingFunc);



        // 910C 对应 CANN 的 910_93
        this->AICore()
            .AddConfig(
                "ascend910_93");

    }

};



OP_ADD(SubCustomTemplate);


}