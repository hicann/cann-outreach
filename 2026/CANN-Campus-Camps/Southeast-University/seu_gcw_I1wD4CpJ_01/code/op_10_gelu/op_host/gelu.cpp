//v5
#include <algorithm>
#include <vector>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const auto *desc = context->GetInputDesc(0);
    const auto *shape = context->GetInputShape(0);
    if (desc == nullptr || shape == nullptr || context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto dtype = desc->GetDataType();
    if (dtype != ge::DT_FLOAT && dtype != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    const int64_t shapeSize = shape->GetStorageShape().GetShapeSize();
    if (shapeSize <= 0) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t length = static_cast<uint64_t>(shapeSize);
    const uint32_t elementBytes = dtype == ge::DT_FLOAT ? 4 : 2;
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const auto plan = PlanGeluBlocks(length, elementBytes, platform.GetCoreNumAiv());
    uint64_t ubBytes = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);

    // 在完整预算内选择较大的 tile，减少循环和高阶 API 调用次数。
    const uint32_t limit = static_cast<uint32_t>(
        std::min<uint64_t>(plan.blockFactor, GELU_MAX_TILE_ELEMENTS));
    uint32_t tile = 0;
    uint32_t scratchBytes = 0;
    uint32_t low = 1;
    uint32_t high = (limit + 63) / 64;
    while (low <= high) {
        const uint32_t units = low + (high - low) / 2;
        const uint32_t candidate = units * 64;
        uint32_t maximum = 0;
        uint32_t minimum = 0;
        const ge::Shape erfcShape(std::vector<int64_t>{static_cast<int64_t>(candidate)});
        AscendC::GetErfcMaxMinTmpSize(erfcShape, sizeof(float), false, maximum, minimum);
        const uint32_t scratch = (std::max<uint32_t>(maximum, 32) + 31) / 32 * 32;
        // fp32 直接计算到输出队列：只需两个 float 工作区；fp16 仍需三个。
        const uint32_t workBuffers = dtype == ge::DT_FLOAT ? 2 : 3;
        const uint64_t buffers = uint64_t(candidate) *
            (4 * elementBytes + workBuffers * sizeof(float));
        const uint32_t maskBytes = ((candidate + 255) / 256) * 32;
        // 除 Erfc 外，也为 Select 的内部临时空间和运行时保留余量。
        if (buffers + maskBytes + scratch + 16384 <= ubBytes) {
            tile = candidate;
            scratchBytes = scratch;
            low = units + 1;
        } else {
            high = units - 1;
        }
    }
    if (tile == 0) {
        return ge::GRAPH_FAILED;
    }
    auto *tiling = context->GetTilingData<GeluTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->length = length;
    tiling->blockFactor = plan.blockFactor;
    tiling->ubFactor = tile;
    tiling->erfcTmpBytes = scratchBytes;
    // 与 Kernel 中的 TILING_KEY_IS 分支一一对应，不依赖入口模板实例化。
    context->SetTilingKey(dtype == ge::DT_FLOAT16 ? GELU_TILING_FP16 : GELU_TILING_FP32);
    context->SetBlockDim(plan.blockDim);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const auto *input = context->GetInputShape(0);
    auto *output = context->GetOutputShape(0);
    if (input == nullptr || output == nullptr) {
        return GRAPH_FAILED;
    }
    *output = *input;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Gelu : public OpDef {
public:
    explicit Gelu(const char *name) : OpDef(name)
    {
        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910b");
    }
};
OP_ADD(Gelu);
}  // namespace ops


