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
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

static const gert::Shape g_vec_1_shape = {1};

static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& inShape)
{
    if (inShape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return inShape;
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

    coreNum = ascendcPlatform.GetCoreNumAiv();

    OP_CHECK_IF(
        coreNum <= 0,
        OP_LOGE(context, "coreNum is invalid"),
        return ge::GRAPH_FAILED);

    ascendcPlatform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    OP_CHECK_IF(
        ubSize == 0,
        OP_LOGE(context, "UB size is zero"),
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
        OP_LOGE(context, "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    const gert::StorageShape* inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    const gert::Shape& shape =
        EnsureNotScalar(
            inputShape->GetStorageShape());

    int64_t totalNum = 1;

    for (int64_t i = 0; i < shape.GetDimNum(); ++i) {
        totalNum *= shape.GetDim(i);
    }

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "Input element number is invalid"),
        return ge::GRAPH_FAILED);

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    int64_t typeSize = TYPE_SIZE;

    if (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
        inputDesc->GetDataType() == ge::DT_BF16) {
        typeSize = 2;
    }

    /*
     * 输入队列和输出队列均使用双缓冲：
     *
     * input : 2 * ubFactor * sizeof(T)
     * output: 2 * ubFactor * sizeof(T)
     */
    constexpr int64_t BUFFER_NUM = 2;
    constexpr int64_t BUFFER_COUNT = 2 * BUFFER_NUM;

    int64_t ubBlockSize =
        GetUbBlockSize(context);

    if (ubBlockSize <= 0) {
        ubBlockSize = 32;
    }

    int64_t alignElements =
        ubBlockSize / typeSize;

    if (alignElements <= 0) {
        alignElements = 1;
    }

    int64_t ubFactor = static_cast<int64_t>(
        ubSize /
        static_cast<uint64_t>(
            typeSize * BUFFER_COUNT));

    ubFactor =
        FloorAlign(
            ubFactor,
            alignElements);

    if (ubFactor < alignElements) {
        ubFactor = alignElements;
    }

    /*
     * 根据数据量决定使用的核数。
     */
    int64_t usedCoreNum = coreNum;

    if (totalNum <
        coreNum * MIN_SPLIT_THRESHOLD) {
        usedCoreNum =
            CeilDiv(
                totalNum,
                MIN_SPLIT_THRESHOLD);

        if (usedCoreNum < 1) {
            usedCoreNum = 1;
        }

        if (usedCoreNum > coreNum) {
            usedCoreNum = coreNum;
        }
    }

    /*
     * 每个核处理连续的一段数据。
     * 最后一个核通过 kernel 端裁剪实际长度。
     */
    int64_t blockFactor =
        CeilDiv(
            totalNum,
            usedCoreNum);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(
        static_cast<uint32_t>(
            usedCoreNum));

    uint64_t tilingKey = 0;

    if (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
        inputDesc->GetDataType() == ge::DT_BF16) {
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
    [[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(
        TilingParseForRelu);

} // namespace optiling