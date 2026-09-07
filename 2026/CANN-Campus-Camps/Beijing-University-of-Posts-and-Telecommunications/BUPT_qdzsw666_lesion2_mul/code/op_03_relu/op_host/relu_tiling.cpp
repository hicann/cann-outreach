/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include <algorithm>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"

#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::FloorAlign;

constexpr uint32_t WS_SYS_SIZE = 0U;

// 按最大的 float 类型计算，避免 UB 空间分配过大
constexpr int64_t TYPE_SIZE = 4;

// 一个 DataBlock 为 32 Byte
constexpr int64_t BLOCK_SIZE = 32;

// 输入数据较小时，只启动一个核
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;

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
        coreNum <= 0,
        OP_LOGE(context, "coreNum is invalid"),
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

    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    currentWorkspace[0] = WS_SYS_SIZE;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum)
            != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo failed"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize failed"),
        return ge::GRAPH_FAILED);

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 获取输入形状
    const gert::StorageShape* inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const gert::Shape& storageShape =
        inputShape->GetStorageShape();

    // 计算输入Tensor的总元素数
    int64_t totalNum = 1;

    for (int64_t i = 0;
         i < storageShape.GetDimNum();
         ++i) {
        totalNum *= storageShape.GetDim(i);
    }

    // 防止非法或空输入
    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "totalNum must be greater than 0"),
        return ge::GRAPH_FAILED);

    /*
     * 根据数据量确定实际使用的核数。
     *
     * 每个核尽量至少处理 MIN_SPLIT_THRESHOLD 个元素，
     * 同时不能超过设备可用的AI Vector Core数量。
     */
    int64_t usedCoreNum =
        CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);

    usedCoreNum = std::max<int64_t>(usedCoreNum, 1);
    usedCoreNum = std::min<int64_t>(usedCoreNum, coreNum);

    // 每个核最多处理的元素数
    int64_t blockFactor =
        CeilDiv(totalNum, usedCoreNum);

    /*
     * UB中同时存在：
     *   inputQueueX：BUFFER_NUM个buffer
     *   outputQueueY：BUFFER_NUM个buffer
     *
     * 所以一共需要：
     *   2 × BUFFER_NUM × ubFactor × TYPE_SIZE
     *
     * BUFFER_NUM固定为2，因此这里除以4。
     */
    constexpr int64_t BUFFER_NUM = 2;
    constexpr int64_t TENSOR_NUM = 2;

    int64_t ubFactor =
        static_cast<int64_t>(ubSize) /
        (BUFFER_NUM * TENSOR_NUM * TYPE_SIZE);

    // 以32 Byte对齐；按float计算即8个元素对齐
    const int64_t alignNum = BLOCK_SIZE / TYPE_SIZE;

    ubFactor = FloorAlign(ubFactor, alignNum);

    OP_CHECK_IF(
        ubFactor <= 0,
        OP_LOGE(context, "UB space is insufficient"),
        return ge::GRAPH_FAILED);

    // Tile长度不需要超过单核数据长度
    ubFactor = std::min<int64_t>(ubFactor, blockFactor);

    // 设置Tiling数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 设置实际启动核数
    context->SetBlockDim(usedCoreNum);

    // 根据输入dtype选择Tiling Key
    uint64_t tilingKey = 0;

    const gert::CompileTimeTensorDesc* inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    if (inputDesc->GetDataType() == ge::DT_FLOAT16 ||
        inputDesc->GetDataType() == ge::DT_BF16) {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
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
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling