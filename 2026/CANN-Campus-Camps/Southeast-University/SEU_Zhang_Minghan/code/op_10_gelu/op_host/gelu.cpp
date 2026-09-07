// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    int32_t num_cores_aiv = platform.GetCoreNumAiv();
    if (num_cores_aiv <= 0) {
        num_cores_aiv = 1;
    }
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

    // 获取算子输入信息
    const gert::Tensor *tensor_input_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_input_x = tensor_input_x->GetDataType();
    int32_t dtype_size_input_x = ge::GetSizeByDataType(dtype_input_x);
    uint32_t length_input_x = tensor_input_x->GetShapeSize();  // 元素个数

    // 配置 tiling key, 从而区分 kernel 侧 float32 / float16 两种模板实例
    uint32_t DT_INPUT_X = static_cast<uint32_t>(dtype_input_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_INPUT_X);

    // 填充 tiling 结构体
    GeluTilingData *tiling = context->GetTilingData<GeluTilingData>();
    tiling->length = length_input_x;

    // 一个数据块 = 32B 对齐, 换算为元素个数: float32 -> 8, float16 -> 16
    constexpr uint32_t kBlockBytes = 32;
    const uint32_t elem_per_block = kBlockBytes / static_cast<uint32_t>(dtype_size_input_x);

    // 总元素个数向上取整到 32B 对齐, 并按 32B 块计数
    const uint32_t aligned_length =
        ((length_input_x + elem_per_block - 1) / elem_per_block) * elem_per_block;
    const uint32_t block_count = aligned_length / elem_per_block;

    if (block_count == 0) {  // 防御: 空张量(题目约束下不会出现)
        tiling->smallCoreDataNum = 0;
        tiling->bigCoreDataNum = 0;
        tiling->finalSmallTileNum = 0;
        tiling->finalBigTileNum = 0;
        tiling->tileDataNum = elem_per_block;
        tiling->smallTailDataNum = 0;
        tiling->bigTailDataNum = 0;
        tiling->tailBlockNum = 0;
        context->SetBlockDim(1);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }

    // 按实际需要的块数自适应启动核数: 小张量避免开大量空转核, 大张量用满全部 AIV
    const uint32_t active_cores =
        std::min(static_cast<uint32_t>(num_cores_aiv), block_count);

    // 将 32B 块尽量均分到各核, 剩余块交给前 extra_blocks 个核(大核)
    const uint32_t base_blocks_per_core = block_count / active_cores;
    const uint32_t extra_blocks = block_count % active_cores;
    const uint32_t small_core_data_num = base_blocks_per_core * elem_per_block;
    const uint32_t big_core_data_num = (base_blocks_per_core + 1) * elem_per_block;
    const uint32_t max_core_data =
        (extra_blocks > 0) ? big_core_data_num : small_core_data_num;

    // 依据 UB 空间计算单个 tile 的最大元素个数:
    // 输入/输出各 2 个 buffer 双缓冲, 共 4 份 tile, 并预留约 1/4 UB 余量
    constexpr uint32_t kBufferFactor = 4;
    const uint32_t available_ub = static_cast<uint32_t>(ub_size) * 3 / 4;
    uint32_t tile_cap =
        available_ub / (kBufferFactor * static_cast<uint32_t>(dtype_size_input_x));
    tile_cap = std::max((tile_cap / elem_per_block) * elem_per_block, elem_per_block);

    // 每个核的数据尽量拆成 2 个以上 tile, 以发挥双缓冲流水(DMA 与向量计算重叠);
    // 单核数据不足 2 个 32B 块时整块一次处理。
    const uint32_t half_core = (max_core_data / elem_per_block / 2) * elem_per_block;
    uint32_t tile_data_num = 0;
    if (half_core >= elem_per_block) {
        tile_data_num = std::min(tile_cap, half_core);
    } else {
        tile_data_num = max_core_data;
    }
    tile_data_num = std::max(tile_data_num, elem_per_block);
    tile_data_num = std::min(tile_data_num, max_core_data);

    tiling->smallCoreDataNum = small_core_data_num;
    tiling->bigCoreDataNum = big_core_data_num;
    tiling->finalSmallTileNum = (small_core_data_num == 0)
                                    ? 0
                                    : (small_core_data_num + tile_data_num - 1) / tile_data_num;
    tiling->finalBigTileNum = (big_core_data_num == 0)
                                  ? 0
                                  : (big_core_data_num + tile_data_num - 1) / tile_data_num;
    tiling->smallTailDataNum = (small_core_data_num % tile_data_num == 0)
                                   ? tile_data_num
                                   : (small_core_data_num % tile_data_num);
    tiling->bigTailDataNum = (big_core_data_num % tile_data_num == 0)
                                 ? tile_data_num
                                 : (big_core_data_num % tile_data_num);
    tiling->tileDataNum = tile_data_num;
    tiling->tailBlockNum = extra_blocks;

    // 配置启动核数
    context->SetBlockDim(active_cores);
    // 配置 workspace 大小(本算子无需额外 workspace)
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context) {
    // 输出 shape 与输入 shape 保持一致
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    *outputShape = *inputShape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    // 输出数据类型与输入数据类型保持一致
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