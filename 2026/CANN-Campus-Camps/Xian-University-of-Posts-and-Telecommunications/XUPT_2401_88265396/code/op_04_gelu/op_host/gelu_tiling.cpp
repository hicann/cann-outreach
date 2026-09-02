/*!
 * \file gelu_tiling.cpp
 * \brief Gelu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/gelu_tiling_data.h"
#include "../op_kernel/gelu_tiling_key.h"

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

static inline const gert::Shape EnsureNotScalar(const gert::Shape& in_shape)
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

    coreNum = ascendcPlatform.GetCoreNumAiv();

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

static ge::graphStatus GeluTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    GeluTilingData* tiling =
        context->GetTilingData<GeluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    /*
     * 获取实际输入 shape。
     */
    const auto* inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    const gert::Shape storageShape =
        EnsureNotScalar(
            inputShape->GetStorageShape());

    const int64_t totalNum =
        storageShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "invalid totalNum"),
        return ge::GRAPH_FAILED);

    /*
     * 获取输入 dtype。
     */
    const auto* inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    const ge::DataType dataType =
        inputDesc->GetDataType();

    int64_t typeSize = TYPE_SIZE;

    if (dataType == ge::DT_FLOAT16 ||
        dataType == ge::DT_BF16) {
        typeSize = 2;
    }

    /*
     * Ascend 数据搬运基本块为 32 Byte。
     *
     * float32: 8 elements
     * float16: 16 elements
     */
    constexpr int64_t BLOCK_BYTES = 32;

    const int64_t alignNum =
        BLOCK_BYTES / typeSize;

    /*
     * 根据数据量选择 Core 数。
     *
     * 本题：
     *
     * 8 * 2048 = 16384
     *
     * 通常会选择 16 个 Core，
     * 每 Core 处理 1024 个元素。
     */
    int64_t usedCoreNum =
        (totalNum +
         MIN_SPLIT_THRESHOLD - 1) /
        MIN_SPLIT_THRESHOLD;

    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }

    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    /*
     * 保证：
     *
     * 1. totalNum 可以被 Core 数整除
     * 2. 每个 Core 的数据量满足 32B 对齐
     */
    while (usedCoreNum > 1) {
        const bool divisible =
            (totalNum % usedCoreNum) == 0;

        if (divisible) {
            const int64_t perCore =
                totalNum / usedCoreNum;

            if ((perCore % alignNum) == 0) {
                break;
            }
        }

        --usedCoreNum;
    }

    const int64_t blockFactor =
        totalNum / usedCoreNum;

    /*
     * GELU 的 Erf 属于复杂数学计算，
     * 内部还需要额外临时 UB。
     *
     * 因此这里不像简单 Relu 那样尽可能使用整个 UB，
     * 而是保守地把单 Tile 控制为最多 1024 个元素。
     *
     * 这样：
     *
     * input double buffer
     * output double buffer
     * tmp buffer
     * Erf 内部临时空间
     *
     * 都有足够 UB。
     */
    constexpr int64_t MAX_UB_FACTOR = 1024;

    int64_t ubFactor =
        blockFactor < MAX_UB_FACTOR
            ? blockFactor
            : MAX_UB_FACTOR;

    /*
     * 保证 DataCopy 长度满足 32B 对齐。
     */
    ubFactor =
        (ubFactor / alignNum) *
        alignNum;

    if (ubFactor <= 0) {
        ubFactor = alignNum;
    }

    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }

    /*
     * 写入 TilingData。
     */
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(
        static_cast<uint32_t>(
            usedCoreNum));

    /*
     * dtype -> TilingKey
     *
     * mode 0 : half
     * mode 1 : float
     */
    uint64_t tilingKey;

    if (dataType == ge::DT_FLOAT16 ||
        dataType == ge::DT_BF16) {
        tilingKey =
            GET_TPL_TILING_KEY(
                GELU_TPL_SCH_MODE_0);
    } else {
        tilingKey =
            GET_TPL_TILING_KEY(
                GELU_TPL_SCH_MODE_1);
    }

    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForGelu(
    [[maybe_unused]]
    gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct GeluCompileInfo {};

IMPL_OP_OPTILING(Gelu)
    .Tiling(GeluTilingFunc)
    .TilingParse<GeluCompileInfo>(
        TilingParseForGelu);

} // namespace optiling