#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "../op_kernel/tanh_custom_tiling.h"

namespace optiling {


static ge::graphStatus TilingFunc(
    gert::TilingContext* context)
{

    uint32_t totalLength =
        context->GetInputShape(0)
        ->GetOriginShape()
        .GetShapeSize();



    auto tiling =
        context->GetRawTilingData();



    TanhCustomTilingData* data =
        reinterpret_cast<TanhCustomTilingData*>(
            tiling->GetData()
        );



    data->totalLength = totalLength;


    data->tileNum = 8;


    tiling->SetDataSize(
        sizeof(TanhCustomTilingData)
    );



    context->SetBlockDim(8);


    return ge::GRAPH_SUCCESS;

}

}



namespace ops {


class TanhCustom : public OpDef {


public:

    explicit TanhCustom(
        const char* name)
        : OpDef(name)
    {


        this->Input("x")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});


        this->Output("y")
        .ParamType(REQUIRED)
        .DataType({ge::DT_FLOAT16})
        .Format({ge::FORMAT_ND});


	this->SetInferShape([](gert::InferShapeContext* context) {
    auto input_shape = context->GetInputShape(0);
    auto output_shape = context->GetOutputShape(0);

    *output_shape = *input_shape;

    return ge::GRAPH_SUCCESS;
});


this->SetInferDataType([](gert::InferDataTypeContext* context) {

    auto input_dtype =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        input_dtype
    );

    return ge::GRAPH_SUCCESS;
});

        this->AICore()
        .SetTiling(
            optiling::TilingFunc
        )
        .AddConfig(
            "ascend910b"
        );

    }


};


OP_ADD(TanhCustom);


}