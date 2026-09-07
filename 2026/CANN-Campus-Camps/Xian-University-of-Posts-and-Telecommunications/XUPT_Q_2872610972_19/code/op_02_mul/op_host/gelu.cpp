// Host侧Tiling实现: Gelu
// 输出z的shape、dtype与输入x一致(支持float16/float32, ND格式)。
// tiling方案: 按32B对齐块在全部AIV核上分配, 前tailBlockNum个核多处理一个
// 对齐块, 其余核处理较少的对齐块; 每核内部按tileDataNum分片搬入/计算/搬出。
#include "register/op_def_registry.h"

#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv <= 0) {
        num_cores_aiv = 1;
    }

    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_input_x = tensor_input_x->GetDataType();
    int32_t dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);
    uint32_t length_input_x = tensor_input_x->GetShapeSize();

    // 配置tiling key, 区分float16/float32
    uint32_t dt_input_x = static_cast<uint32_t>(dtype_input_x);
    ASCENDC_TPL_SEL_PARAM(context, dt_input_x);

    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->length = length_input_x;

    // 配置启动核数与workspace大小(本算子无需额外workspace)
    context->SetBlockDim(num_cores_aiv);
    size_t *workspace_sizes = context->GetWorkspaceSizes(1);
    workspace_sizes[0] = 0;

    // 按32B对齐块分配: 每个块32个元素(与dtype无关按字节对齐, 元素数向上取整到32B)
    constexpr uint32_t kBlockSize = 32;
    const uint32_t aligned_length = ((length_input_x + kBlockSize - 1) / kBlockSize) * kBlockSize;
    const uint32_t block_count = aligned_length / kBlockSize;
    const uint32_t base_blocks_per_core = block_count / num_cores_aiv;
    const uint32_t extra_blocks = block_count % num_cores_aiv;

    const uint32_t small_core_data_num = base_blocks_per_core * kBlockSize;
    const uint32_t big_core_data_num = (base_blocks_per_core + 1) * kBlockSize;

    // 根据UB空间确定单次tile的元素个数(留出0.75*UB, 且不超过输入长度、不小于一个对齐块)
    uint32_t available_ub = static_cast<uint32_t>(ub_size) * 3 / 4;
    uint32_t tile_data_num = available_ub / (2 * 2 * static_cast<uint32_t>(dtype_size_input_x));
    if (tile_data_num < kBlockSize) {
        tile_data_num = kBlockSize;
    }
    tile_data_num = std::min(tile_data_num, aligned_length);
    if (tile_data_num == 0) {
        tile_data_num = kBlockSize;
    }
    tile_data_num = ((tile_data_num + kBlockSize - 1) / kBlockSize) * kBlockSize;

    tiling->smallCoreDataNum = small_core_data_num;
    tiling->bigCoreDataNum = big_core_data_num;
    tiling->finalSmallTileNum = (small_core_data_num == 0) ? 0 : ((small_core_data_num + tile_data_num - 1) / tile_data_num);
    tiling->finalBigTileNum = (big_core_data_num == 0) ? 0 : ((big_core_data_num + tile_data_num - 1) / tile_data_num);
    tiling->smallTailDataNum = (small_core_data_num % tile_data_num == 0) ? tile_data_num : (small_core_data_num % tile_data_num);
    tiling->bigTailDataNum = (big_core_data_num % tile_data_num == 0) ? tile_data_num : (big_core_data_num % tile_data_num);
    tiling->tileDataNum = tile_data_num;
    tiling->tailBlockNum = extra_blocks;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    // 输出shape与输入x一致
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    // 输出dtype与输入x一致
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name) {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}  // namespace ops
