#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

// 硬件常量（Ascend 910B）
constexpr uint32_t UB_SIZE = 256 * 1024;           
constexpr uint32_t MIN_BLOCK_SIZE = 16 * 1024;          
constexpr uint32_t MAX_BLOCK_DIM = 40; // 910B 最大核数
constexpr uint32_t BUFFER_NUM = 2;                  // Double Buffer
constexpr uint32_t RESERVED = 0;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
  AddCustomTemplateTilingData *tiling = context->GetTilingData<AddCustomTemplateTilingData>();
  //获取数据大小/类型  
  uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
  auto dataType = context->GetInputDesc(0)->GetDataType();
  uint32_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;
  //如果为空直接跳  
    if (totalLength == 0) {
        tiling->totalLength = 0;
        tiling->blockDim = 1;
        tiling->tileNum = 0;
        tiling->tileLength = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }
    uint32_t alignNum = 32 / typeSize;  
    

    //blockDim
    uint32_t blockDim = totalLength / MIN_BLOCK_SIZE;  // 每个核至少处理 MIN_BLOCK_SIZE 个元素
    if (blockDim == 0) blockDim = 1;
    if (blockDim > MAX_BLOCK_DIM) blockDim = MAX_BLOCK_DIM;
    
    //blockLength
    uint32_t blockLength = totalLength / blockDim;
    blockLength = ((blockLength-1) / alignNum+1) * alignNum;//向上取整，对齐
 
    //tileNum
    uint32_t tileNum=8;

    //tileLength
    uint32_t maxTileLength =(UB_SIZE - RESERVED) / typeSize/BUFFER_NUM;
    maxTileLength = ((maxTileLength -1)/ alignNum+1) * alignNum;//向上取整
    uint32_t tileLength = std::min(blockLength / tileNum, maxTileLength);

    //re-tileNum
    tileNum=(blockLength-1)/tileLength+1;

    context->SetBlockDim(blockDim);
    tiling->BlockLength = blockLength;
    tiling->tileNum = tileNum;
    tiling->tileLength = tileLength;
    tiling->blockDim = blockDim;
    tiling->typeSize = typeSize;
    tiling->totalLength = totalLength;

    
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
class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char* name) : OpDef(name)
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
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};
OP_ADD(AddCustomTemplate);
}
