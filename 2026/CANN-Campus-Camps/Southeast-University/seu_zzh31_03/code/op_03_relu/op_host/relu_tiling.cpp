/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <algorithm>
#include <limits>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// 单次搬入搬出通道的 buffer 个数（对应 op_kernel/relu.h 中的 BUFFER_NUM）
constexpr int64_t BUFFER_NUM = 2;
// 32B 对齐：DataCopy 要求 count*sizeof(T) 为 32B 对齐
constexpr int64_t BLOCK_ALIGN_BYTES = 32;
// 单次 vector 指令处理的安全 tile 元素数上限（参考 add_custom 模板的 TILE_LENGTH）
constexpr int64_t MAX_TILE_SIZE = 4096;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape) {
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }
    return in_shape;
}

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, uint64_t& ubSize, int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 1. 计算输入张量总元素数量（支持标量 shape）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    auto inputShape = context->GetInputShape(0);
    int64_t totalNum = 1;
    if (inputShape != nullptr) {
        totalNum = inputShape->GetShape().GetShapeSize();
    }
    if (totalNum <= 0) {
        OP_LOGE(context, "input shape size is invalid");
        return ge::GRAPH_FAILED;
    }

    // 2. 数据类型与单元素字节数
    ge::DataType dtype = inputDesc->GetDataType();
    int64_t typeSize = TYPE_SIZE;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        typeSize = 2;
    }

    // 3. 参与计算的核心数量：不超过总元素数，保证每核至少有 1 个元素
    int64_t blockDim = std::min<int64_t>(coreNum, totalNum);
    if (blockDim <= 0) {
        OP_LOGE(context, "blockDim is invalid");
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 4. 每个核连续处理的元素个数。
    //    向上 32B 对齐，保证本核输入/输出在 GM 的起始地址 32B 对齐（DataCopyPad/DataCopy 要求）。
    int64_t alignElem = BLOCK_ALIGN_BYTES / typeSize;  // float: 8, half: 16
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, blockDim), alignElem);

    // 5. 单次 UB 循环处理的元素个数（ubFactor）。
    //    考虑 DoubleBuffer：VECIN(BUFFER_NUM) + VECOUT(BUFFER_NUM) 共 2*BUFFER_NUM 个常驻 UB buffer，
    //    故单 buffer 元素数上限为 ubSize / (2 * BUFFER_NUM * typeSize)。
    int64_t maxByUb = static_cast<int64_t>(ubSize) / (2 * BUFFER_NUM * typeSize);
    int64_t maxByCopy = std::numeric_limits<uint16_t>::max() / typeSize;  // DataCopyPad blockLen 上限
    int64_t maxTile = FloorAlign(std::min<int64_t>({maxByUb, maxByCopy, MAX_TILE_SIZE}), alignElem);
    int64_t ubFactor = std::min<int64_t>(blockFactor, maxTile);
    if (ubFactor <= 0) {
        ubFactor = alignElem;
    }

    // 6. 写入 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 根据输入 dtype 选择 tilingKey（MODE_0: half, MODE_1: float）
    uint64_t tilingKey;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
