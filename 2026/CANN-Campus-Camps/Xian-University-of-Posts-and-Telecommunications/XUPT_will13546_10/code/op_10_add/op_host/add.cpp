// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/add_tiling.h"
#include "../op_kernel/tiling_key_add.h"

namespace optiling {
namespace {
constexpr uint32_t TILE_LENGTH = 2048;  // 每块最多元素数，仅用于Host侧估算tileNum
}

static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv == 0) {
        num_cores_aiv = 1;
    }

    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    if (tensor_x == nullptr) {
        return ge::GRAPH_FAILED;
    }

    uint32_t total_length = tensor_x->GetShapeSize();
    uint32_t block_dim = num_cores_aiv;
    if (total_length == 0) {
        block_dim = 1;
    } else if (total_length < block_dim) {
        block_dim = total_length;
    }

    // 按输入shape估算每个核上的块数
    uint32_t block_length = (total_length + block_dim - 1) / block_dim;
    uint32_t tile_num = (block_length + TILE_LENGTH - 1) / TILE_LENGTH;
    if (tile_num == 0) {
        tile_num = 1;
    }

    // 配置TilingKey，按输入数据类型选择kernel模板
    uint32_t dt_x = static_cast<uint32_t>(tensor_x->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dt_x);

    AddTilingData *tiling = context->GetTilingData<AddTilingData>();
    tiling->totalLength = total_length;
    tiling->tileNum = tile_num;

    context->SetBlockDim(block_dim);

    size_t *current_workspace = context->GetWorkspaceSizes(1);
    current_workspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape *input_shape = context->GetInputShape(0);
    gert::Shape *output_shape = context->GetOutputShape(0);
    if (input_shape == nullptr || output_shape == nullptr) {
        return GRAPH_FAILED;
    }
    *output_shape = *input_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Add : public OpDef {
public:
    explicit Add(const char *name) : OpDef(name) {
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
OP_ADD(Add);
}  // namespace ops