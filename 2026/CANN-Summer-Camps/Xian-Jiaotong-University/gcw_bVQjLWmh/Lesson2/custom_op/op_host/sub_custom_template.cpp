#include "../op_kernel/sub_custom_template_tiling.h"
 	 #include "register/op_def_registry.h"
 	 
 	 
 	 namespace optiling {
 	 static ge::graphStatus TilingFunc(gert::TilingContext* context)
 	 {
 	 
 	   SubCustomTemplateTilingData *tiling = context->GetTilingData<SubCustomTemplateTilingData>();
    const gert::StorageShape* x1_shape = context->GetInputShape(0);
    if (x1_shape == nullptr) {
      return ge::GRAPH_FAILED;
    }
 	   int32_t data_sz = 1;
 	   for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
 	     data_sz *= x1_shape->GetStorageShape().GetDim(i);
    int64_t data_sz = 1;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
      data_sz *= x1_shape->GetStorageShape().GetDim(i);
    tiling->size = data_sz;
  return ge::GRAPH_SUCCESS;
    for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
      data_sz *= x1_shape->GetStorageShape().GetDim(i);
  return ge::GRAPH_SUCCESS;
}
}  // namespace optiling