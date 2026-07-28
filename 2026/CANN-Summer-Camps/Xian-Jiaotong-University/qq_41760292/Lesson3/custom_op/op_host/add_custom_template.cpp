#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"


namespace optiling {


    static ge::graphStatus TilingFunc(
        gert::TilingContext* context)
    {

        AddCustomTemplateTilingData* tiling =
            context->GetTilingData<
            AddCustomTemplateTilingData>();



        /*
         * CANN 9.0.0:
         * GetInputShape返回StorageShape
         */
        const gert::StorageShape* inputShape =
            context->GetInputShape(0);



        const uint32_t totalLength =
            inputShape->GetStorageShape()
            .GetShapeSize();



        /*
         * 保持测试通过配置
         */
        context->SetBlockDim(40);



        tiling->totalLength =
            totalLength;



        return ge::GRAPH_SUCCESS;
    }

}



namespace ge {


    static ge::graphStatus InferShape(
        gert::InferShapeContext* context)
    {

        const gert::Shape* inputShape =
            context->GetInputShape(0);


        gert::Shape* outputShape =
            context->GetOutputShape(0);



        *outputShape =
            *inputShape;



        return GRAPH_SUCCESS;
    }



    static ge::graphStatus InferDataType(
        gert::InferDataTypeContext* context)
    {


        const auto inputType =
            context->GetInputDataType(0);



        context->SetOutputDataType(
            0,
            inputType);



        return GRAPH_SUCCESS;

    }

}



namespace ops {


    class AddCustomTemplate :
        public OpDef {


    public:


        explicit AddCustomTemplate(
            const char* name)
            : OpDef(name)
        {


            this->Input("x")
                .ParamType(REQUIRED)
                .DataType(
                    {
                        ge::DT_FLOAT16,
                        ge::DT_FLOAT
                    })
                .Format(
                    {
                        ge::FORMAT_ND,
                        ge::FORMAT_ND
                    })
                .UnknownShapeFormat(
                    {
                        ge::FORMAT_ND,
                        ge::FORMAT_ND
                    });



            this->Input("y")
                .ParamType(REQUIRED)
                .DataType(
                    {
                        ge::DT_FLOAT16,
                        ge::DT_FLOAT
                    })
                .Format(
                    {
                        ge::FORMAT_ND,
                        ge::FORMAT_ND
                    })
                .UnknownShapeFormat(
                    {
                        ge::FORMAT_ND,
                        ge::FORMAT_ND
                    });



            this->Output("z")
                .ParamType(REQUIRED)
                .DataType(
                    {
                        ge::DT_FLOAT16,
                        ge::DT_FLOAT
                    })
                .Format(
                    {
                        ge::FORMAT_ND,
                        ge::FORMAT_ND
                    })
                .UnknownShapeFormat(
                    {
                        ge::FORMAT_ND,
                        ge::FORMAT_ND
                    });



            this->SetInferShape(
                ge::InferShape)
                .SetInferDataType(
                    ge::InferDataType);



            auto& aiCore =
                this->AICore();



            aiCore.SetTiling(
                optiling::TilingFunc);



            aiCore.AddConfig(
                "ascend910_93");


            aiCore.AddConfig(
                "ascend910b");

        }

    };


    OP_ADD(AddCustomTemplate);

}