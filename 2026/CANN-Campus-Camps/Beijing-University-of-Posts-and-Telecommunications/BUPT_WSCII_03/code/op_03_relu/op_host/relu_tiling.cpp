/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

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

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return in_shape;
}

static ge::graphStatus GetPlatformInfo(
    gert::TilingContext* context,
    uint64_t& ubSize,
    int64_t& coreNum)
{
    fe::PlatFormInfos* platformInfoPtr =
        context->GetPlatformInfo();

    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(platformInfoPtr);

    coreNum =
        ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum == 0,
        OP_LOGE(context, "coreNum is 0"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "ubSize is 0"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetWorkspaceSize(
    gert::TilingContext* context)
{
    size_t* currentWorkspace =
        context->GetWorkspaceSizes(1);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        currentWorkspace);

    currentWorkspace[0] =
        WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}


// ============================================================
// Relu Tiling
// ============================================================
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

    // ==========================
    // 获取输入元素总数
    // ==========================

    auto inputX = context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputX);

    auto inputShapeX =
        EnsureNotScalar(
            inputX->GetStorageShape());

    int64_t totalNum =
        inputShapeX.GetShapeSize();

    // ==========================
    // 获取 TilingData
    // ==========================

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    /*
     * DataCopy 对齐：
     *
     * FP16:
     * 16 elements = 32B
     *
     * FP32:
     * 16 elements = 64B
     *
     * 所以统一按 16 elements 对齐。
     */
    constexpr int64_t ALIGN_NUM = 16;

    /*
     * 尽量使用全部 AIV。
     *
     * 例如：
     *
     * totalNum = 16384
     * coreNum  = 40
     *
     * ceil(16384 / 40)
     * = 410
     *
     * align(410, 16)
     * = 416
     */
    int64_t blockFactor =
        CeilAlign(
            CeilDiv(totalNum, coreNum),
            ALIGN_NUM);

    /*
     * 重新计算实际需要的核数。
     *
     * 40 AIV:
     * ceil(16384 / 416) = 40
     */
    int64_t usedCoreNum =
        CeilDiv(
            totalNum,
            blockFactor);

    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    /*
     * 一个 Core 整块一次搬入。
     *
     * 不再核内切多个 tile。
     */
    tiling->ubFactor =
        blockFactor;

    context->SetBlockDim(
        static_cast<uint32_t>(
            usedCoreNum));

    // ==========================
    // Tiling Key
    // ==========================

    uint64_t tilingKey;

    auto inputDesc =
        context->GetInputDesc(0);

    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
         inputDesc->GetDataType() == ge::DT_BF16))
    {
        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_0);
    }
    else
    {
        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(
        tilingKey);

    return ge::GRAPH_SUCCESS;
}


static ge::graphStatus TilingParseForRelu(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);

} // namespace optiling