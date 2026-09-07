// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 1. 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    // 2. 获取输入Tensor信息
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t length_x = tensor_x->GetShapeSize();

    // 3. 根据dtype选择对应Kernel模板
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 4. Tiling方案
    // 本题shape固定为(8, 2048)，优先使用8个Vector Core。
    uint32_t block_num = (num_cores_aiv >= 8) ? 8 : static_cast<uint32_t>(num_cores_aiv);
    if (block_num == 0) {
        block_num = 1;
    }
    // 保证能够整除，避免出现尾块。
    while (block_num > 1 && length_x % block_num != 0) {
        --block_num;
    }

    uint32_t block_length = length_x / block_num;
    constexpr uint32_t TILE_LENGTH = 128;
    uint32_t tile_length = TILE_LENGTH;
    if (block_length < TILE_LENGTH) {
        tile_length = block_length;
    }

    // 当前题目的(8,2048)在fp32/fp16下均可被128整除且满足32B对齐。
    uint32_t tile_num = block_length / tile_length;

    MulTilingData *tiling = context->GetTilingData<MulTilingData>();
    tiling->length = length_x;
    tiling->blockLength = block_length;
    tiling->tileLength = tile_length;
    tiling->tileNum = tile_num;

    // 5. 配置启动核数
    context->SetBlockDim(block_num);

    // 6. 本算子不需要额外workspace
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    // z的shape与x一致
    const gert::Shape *x_shape = context->GetInputShape(0);
    gert::Shape *z_shape = context->GetOutputShape(0);
    *z_shape = *x_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    // z的dtype与x一致
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Mul : public OpDef {
public:
    explicit Mul(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Mul);
}  // namespace ops
