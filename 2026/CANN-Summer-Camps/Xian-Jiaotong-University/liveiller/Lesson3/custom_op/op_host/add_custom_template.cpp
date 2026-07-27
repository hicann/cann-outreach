#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"
#include <algorithm>

// 硬件及策略常量定义（面向 Ascend 910B）
constexpr uint32_t UB_SIZE = 256 * 1024;         // Unified Buffer 总大小 (256KB)
constexpr uint32_t MIN_BLOCK_SIZE = 16 * 1024;   // 每个核推荐处理的最小元素量
constexpr uint32_t MAX_BLOCK_DIM = 40;          // Ascend 910B 最大 AI Core 核数
constexpr uint32_t BUFFER_NUM = 2;              // 双缓冲控制机制 (Double Buffer)
constexpr uint32_t RESERVED = 0;                 // 预留内存大小

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<AddCustomTemplateTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 1. 获取输入张量的总元素量与数据类型
    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    auto dataType = context->GetInputDesc(0)->GetDataType();
    uint32_t typeSize = (dataType == ge::DT_FLOAT16) ? 2 : 4;

    // 2. 边界处理：如果数据量为空，直接返回最简 Tiling 配置
    if (totalLength == 0) {
        tiling->totalLength = 0;
        tiling->blockDim = 1;
        tiling->tileNum = 0;
        tiling->tileLength = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    // 3. 计算 32 字节对齐所对应的元素个数（FP16 为 16 个，FP32 为 8 个）
    uint32_t alignNum = 32 / typeSize;  

    // 4. 计算 BlockDim（决定开启多少个 AI Core 参与并行计算）
    uint32_t blockDim = totalLength / MIN_BLOCK_SIZE;
    if (blockDim == 0) {
        blockDim = 1;
    }
    if (blockDim > MAX_BLOCK_DIM) {
        blockDim = MAX_BLOCK_DIM;
    }
    
    // 5. 计算每个核心的基础处理量，并向上进行 32 字节对齐
    uint32_t blockLength = totalLength / blockDim;
    blockLength = ((blockLength + alignNum - 1) / alignNum) * alignNum;

    // 6. 计算单次切块的极限长度 (maxTileLength)
    uint32_t defaultTileNum = 8;
    uint32_t maxTileLength = (UB_SIZE - RESERVED) / typeSize / BUFFER_NUM;
    maxTileLength = ((maxTileLength + alignNum - 1) / alignNum) * alignNum; 

    // 7. 结合默认切块数与 UB 限制，得出最终的单次循环切块长度
    uint32_t tileLength = std::min(blockLength / defaultTileNum, maxTileLength);

    // 8. 重新计算当前核心实际需要的总循环切块次数 (tileNum)
    uint32_t tileNum = (blockLength + tileLength - 1) / tileLength;

    // 9. 将计算好的 Tiling 结构体参数保存并回传给框架
    context->SetBlockDim(blockDim);
    tiling->BlockLength = blockLength;
    tiling->tileNum = tileNum;
    tiling->tileLength = tileLength;
    tiling->blockDim = blockDim;
    tiling->typeSize = typeSize;
    tiling->totalLength = totalLength;
    
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {

// 形状推导：输出形状与输入 x1 保持一致
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x1_shape != nullptr && y_shape != nullptr) {
        *y_shape = *x1_shape;
    }
    return GRAPH_SUCCESS;
}

// 类型推导：输出数据类型与输入保持一致
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}

} // namespace ge

namespace ops {

// 算子原型注册与配置
class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char* name) : OpDef(name)
    {
        // 注册输入 X
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // 注册输入 Y
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // 注册输出 Z
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // 绑定推导流与 Tiling 函数
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AddCustomTemplate);

} // namespace ops