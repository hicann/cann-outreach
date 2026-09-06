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

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // ------------------------------------------------------------
    // 1. 获取输入 Shape 和总元素数
    // ------------------------------------------------------------
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const gert::Shape& originShape =
        EnsureNotScalar(inputShape->GetOriginShape());

    int64_t totalNum = originShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "input shape size should be greater than 0"),
        return ge::GRAPH_FAILED);

    // ------------------------------------------------------------
    // 2. 计算 32B block 对应的元素数
    //
    // TYPE_SIZE = 4，故典型情况下：
    //   32 Byte / 4 Byte = 8 elements
    // ------------------------------------------------------------
    const int64_t ubBlockSize = static_cast<int64_t>(GetUbBlockSize(context));
    const int64_t alignNum = ubBlockSize / TYPE_SIZE;

    OP_CHECK_IF(
        alignNum <= 0,
        OP_LOGE(context, "invalid alignNum"),
        return ge::GRAPH_FAILED);

    // ------------------------------------------------------------
    // 3. 确定使用多少个 AIV Core
    //
    // 小于 1024 elements 时不拆核；
    // 否则保证平均每核至少处理约 1024 elements。
    // ------------------------------------------------------------
    int64_t useCoreNum = 1;

    if (totalNum >= MIN_SPLIT_THRESHOLD) {
        useCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
        useCoreNum = std::min(useCoreNum, coreNum);
    }

    useCoreNum = std::max<int64_t>(useCoreNum, 1);

    // ------------------------------------------------------------
    // 4. 每个 Core 处理的数据量
    //
    // 向 32B 对应的元素数对齐，方便 DataCopy。
    // 最后一个 Core 可以通过 totalNum 判断实际有效长度。
    // ------------------------------------------------------------
    int64_t blockFactor =
        CeilAlign(CeilDiv(totalNum, useCoreNum), alignNum);

    // 如果对齐后实际上不需要这么多核，重新计算核数，
    // 防止最后出现完全没有数据处理的 Core。
    useCoreNum = CeilDiv(totalNum, blockFactor);
    useCoreNum = std::min(useCoreNum, coreNum);
    useCoreNum = std::max<int64_t>(useCoreNum, 1);

    // ------------------------------------------------------------
    // 5. 计算 UB Factor
    //
    // ReLU 至少需要：
    //     input LocalTensor
    //     output LocalTensor
    //
    // 因此按照两份数据计算。
    //
    // TYPE_SIZE 固定采用 4 Byte，是一个保守方案：
    // FLOAT16/BF16 也可以安全使用。
    // ------------------------------------------------------------
    int64_t maxUbNum =
        static_cast<int64_t>(ubSize) / (2 * TYPE_SIZE);

    maxUbNum = FloorAlign(maxUbNum, alignNum);

    OP_CHECK_IF(
        maxUbNum <= 0,
        OP_LOGE(context, "UB size is too small"),
        return ge::GRAPH_FAILED);

    // 一个 tile 不需要超过一个 core 所负责的数据量
    int64_t ubFactor = std::min(blockFactor, maxUbNum);

    // 保证 tile 满足搬运对齐
    ubFactor = FloorAlign(ubFactor, alignNum);

    // 极小 shape 情况下至少给一个 block
    if (ubFactor == 0) {
        ubFactor = alignNum;
    }

    // ------------------------------------------------------------
    // 6. 写入 TilingData
    // ------------------------------------------------------------
    tiling->totalNum = static_cast<uint32_t>(totalNum);
    tiling->blockFactor = static_cast<uint32_t>(blockFactor);
    tiling->ubFactor = static_cast<uint32_t>(ubFactor);

    context->SetBlockDim(static_cast<uint32_t>(useCoreNum));

    // ------------------------------------------------------------
    // 7. 根据 dtype 选择 kernel 模板
    // ------------------------------------------------------------
    uint64_t tilingKey;

    auto inputDesc = context->GetInputDesc(0);

    if (inputDesc != nullptr &&
        (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
         inputDesc->GetDataType() == ge::DT_BF16)) {
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