// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

// 使用8个AI Core
const uint32_t BLOCK_DIM = 8;

// 每个Core内部的逻辑分块数量
const uint32_t TILE_NUM = 8;


static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    // ========================================================
    // 1. 获取输入Tensor信息
    // ========================================================

    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);


    // ========================================================
    // 2. 获取输入数据类型
    //
    // 用于选择float32 / float16对应的Kernel模板
    // ========================================================

    uint32_t DT_X =
        static_cast<uint32_t>(
            tensor_x->GetDataType());


    // 根据dtype自动配置TilingKey
    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_X);


    // ========================================================
    // 3. 获取TilingData
    // ========================================================

    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();


    // ========================================================
    // 4. 根据输入shape计算总元素数量
    //
    // 本题：
    //
    // shape = (8, 2048)
    //
    // totalLength
    //     = 8 * 2048
    //     = 16384
    // ========================================================

    uint32_t totalLength =
        static_cast<uint32_t>(
            context->GetInputShape(0)
                ->GetOriginShape()
                .GetShapeSize());


    // ========================================================
    // 5. 设置AI Core数量
    // ========================================================

    context->SetBlockDim(BLOCK_DIM);


    // ========================================================
    // 6. 写入Tiling数据
    // ========================================================

    tiling->totalLength = totalLength;

    tiling->tileNum = TILE_NUM;


    // ========================================================
    // 7. Workspace
    //
    // 本算子不需要额外workspace
    // ========================================================

    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;


    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling



// ============================================================
// Shape / DataType推导
// ============================================================

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    // z.shape = x.shape

    const gert::Shape *inputShape =
        context->GetInputShape(0);

    gert::Shape *outputShape =
        context->GetOutputShape(0);

    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}


static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    // z.dtype = x.dtype

    const auto inputDataType =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        inputDataType);

    return ge::GRAPH_SUCCESS;
}

}  // namespace ge



// ============================================================
// 算子注册
// ============================================================

namespace ops {

class Mul : public OpDef {
public:
    explicit Mul(const char *name)
        : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND});

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND});

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops