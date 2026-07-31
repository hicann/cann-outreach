/*!
 * \file truncate_mod_tiling.cpp
 * \brief TruncateMod 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/truncate_mod_tiling_data.h"
#include "../op_kernel/truncate_mod_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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

static ge::graphStatus TruncateModTilingFunc(gert::TilingContext* context)
{
    // TODO: 实现 Tiling 逻辑
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

    TruncateModTilingData* tiling = context->GetTilingData<TruncateModTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ---- 总元素数（取自输入 0 的 shape，逐元素等 shape 场景） ----
    auto x1Shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    const gert::Shape shape = EnsureNotScalar(x1Shape->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum <= 0"), return ge::GRAPH_FAILED);

    // ---- dtype 大小（仅 float32=4，float16/bf16=2） ----
    auto inDesc0 = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inDesc0);
    int64_t dtypeSize = (inDesc0->GetDataType() == ge::DT_FLOAT) ? 4 : 2;

    // 32B 对齐粒度（元素）。
    int64_t ubBlock = static_cast<int64_t>(GetUbBlockSize(context));
    if (ubBlock <= 0) {
        ubBlock = 32;
    }
    int64_t alignNum = ubBlock / dtypeSize;
    if (alignNum <= 0) {
        alignNum = 1;
    }

    // 单核每 tile 上限：3 个 T 队列(double buffer) + 3 个 fp32 计算缓存 + 1 个 int32 缓存，
    // 预留 10% UB 余量。
    int64_t bytesPerElem = 6 * dtypeSize + 3 * static_cast<int64_t>(sizeof(float)) +
                           static_cast<int64_t>(sizeof(int32_t));
    int64_t maxUbNum = FloorAlign(static_cast<int64_t>(ubSize) * 9 / (10 * bytesPerElem), alignNum);
    if (maxUbNum < alignNum) {
        maxUbNum = alignNum;
    }

    // ---- 多核切分：小数据单核，否则按核均分并 32B 对齐。 ----
    if (totalNum <= MIN_SPLIT_THRESHOLD) {
        coreNum = 1;
    }
    int64_t blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignNum);
    if (blockFactor < alignNum) {
        blockFactor = alignNum;
    }
    int64_t usedCore = CeilDiv(totalNum, blockFactor);
    if (usedCore < 1) {
        usedCore = 1;
    }
    int64_t ubFactor = (blockFactor < maxUbNum) ? blockFactor : maxUbNum;
    if (ubFactor < 1) {
        ubFactor = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(usedCore);

    // 根据输入 dtype 选择 tilingKey：
    //   float16 -> MODE_0，float32 -> MODE_1，bfloat16 -> MODE_2
    // （bf16 需单独实例化 TruncateMod<bfloat16_t>，不能当作 half 处理）
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    auto dtype = (inputDesc != nullptr) ? inputDesc->GetDataType() : ge::DT_FLOAT;
    if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_0);
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_2);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTruncateMod([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TruncateModCompileInfo {};

IMPL_OP_OPTILING(TruncateMod).Tiling(TruncateModTilingFunc).TilingParse<TruncateModCompileInfo>(TilingParseForTruncateMod);

} // namespace optiling
