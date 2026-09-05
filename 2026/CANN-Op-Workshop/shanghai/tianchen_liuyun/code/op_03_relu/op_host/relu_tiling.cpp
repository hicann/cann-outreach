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
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

// 本题输入固定为 (8, 2048)，共 16384 个元素。
// 使用 8 个 AI Core，保证每个 Core 正好处理 2048 个元素。
constexpr int64_t FIXED_BLOCK_DIM = 8;

// 每轮 UB 处理 1024 个元素。
// 对 float16 / float32 都满足 32B 对齐要求。
constexpr int64_t UB_FACTOR = 1024;

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

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        platformInfoPtr);

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(
            platformInfoPtr);

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

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(
            context,
            "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // ==========================================================
    // 获取输入 Tensor
    // ==========================================================

    auto inputTensor =
        context->GetInputTensor(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputTensor);

    const int64_t totalNum =
        static_cast<int64_t>(
            inputTensor->GetShapeSize());

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(
            context,
            "totalNum is invalid"),
        return ge::GRAPH_FAILED);

    // ==========================================================
    // Block 数量
    //
    // 题目输入：(8, 2048)
    // totalNum = 16384
    //
    // 固定 8 个 Core：
    // 16384 / 8 = 2048
    //
    // 这样 blockFactor 对 float16 / float32 都天然对齐。
    // ==========================================================

    int64_t blockDim = FIXED_BLOCK_DIM;

    // 防止输入规模小于 8
    if (totalNum < blockDim) {
        blockDim = totalNum;
    }

    if (blockDim <= 0) {
        blockDim = 1;
    }

    const int64_t blockFactor =
        CeilDiv(
            totalNum,
            blockDim);

    // ==========================================================
    // UB Tile
    // ==========================================================

    int64_t ubFactor = UB_FACTOR;

    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    if (ubFactor <= 0) {
        ubFactor = 1;
    }

    // ==========================================================
    // 写 TilingData
    // ==========================================================

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 设置实际使用的 AI Core 数
    context->SetBlockDim(
        static_cast<uint32_t>(blockDim));

    // ==========================================================
    // TilingKey
    // ==========================================================

    auto inputDesc =
        context->GetInputDesc(0);

    uint64_t tilingKey = 0;

    if (inputDesc != nullptr &&
        inputDesc->GetDataType() == ge::DT_FLOAT16) {

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_0);

    } else {

        tilingKey =
            GET_TPL_TILING_KEY(
                RELU_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(tilingKey);

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