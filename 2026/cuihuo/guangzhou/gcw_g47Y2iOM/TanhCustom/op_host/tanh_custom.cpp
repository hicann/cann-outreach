#include "../op_kernel/tanh_custom_tiling.h"
 	 #include "register/op_def_registry.h"
 	 
 	 namespace optiling {
 	 constexpr uint32_t BLOCK_DIM = 8;
 	 constexpr uint32_t TILE_NUM = 8;
 	 
 	 static ge::graphStatus TilingFunc(gert::TilingContext *context)
 	 {
 	     TanhCustomTilingData *tiling =
 	         context->GetTilingData<TanhCustomTilingData>();
 	     tiling->totalLength =
 	         context->GetInputShape(0)->GetOriginShape().GetShapeSize();
 	     tiling->tileNum = TILE_NUM;
 	 
 	     context->SetBlockDim(BLOCK_DIM);
 	     return ge::GRAPH_SUCCESS;
 	 }
 	 }  // namespace optiling
 	 
 	 namespace ge {
 	 static ge::graphStatus InferShape(gert::InferShapeContext *context)
 	 {
 	     const gert::Shape *xShape = context->GetInputShape(0);
 	     gert::Shape *yShape = context->GetOutputShape(0);
 	     *yShape = *xShape;
 	     return GRAPH_SUCCESS;
 	 }
 	 
 	 static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
 	 {
 	     const auto inputDataType = context->GetInputDataType(0);
 	     context->SetOutputDataType(0, inputDataType);
 	     return GRAPH_SUCCESS;
 	 }
 	 }  // namespace ge
 	 
 	 namespace ops {
 	 class TanhCustom : public OpDef {
 	 public:
 	     explicit TanhCustom(const char *name) : OpDef(name)
 	     {
 	         this->Input("x")
 	             .ParamType(REQUIRED)
 	             .DataType({ge::DT_FLOAT16})
 	             .Format({ge::FORMAT_ND})
 	             .UnknownShapeFormat({ge::FORMAT_ND});
 	         this->Output("y")
 	             .ParamType(REQUIRED)
 	             .DataType({ge::DT_FLOAT16})
 	             .Format({ge::FORMAT_ND})
 	             .UnknownShapeFormat({ge::FORMAT_ND});
 	         this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
 	         this->AICore()
 	             .SetTiling(optiling::TilingFunc)
 	             .AddConfig("ascend910b");
 	     }
 	 };
 	 
 	 OP_ADD(TanhCustom);
 	 }  // namespace ops
