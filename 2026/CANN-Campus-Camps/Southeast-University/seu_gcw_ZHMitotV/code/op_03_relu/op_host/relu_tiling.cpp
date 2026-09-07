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
using Ops::Base::FloorDiv;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t BUFFER_NUM = 2;

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


static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    // =========================================================
    // 1. 获取平台信息
    // =========================================================

    uint64_t ubSize;
    int64_t coreNum;

    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);


    // =========================================================
    // 2. Workspace
    // =========================================================

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);


    // =========================================================
    // 3. 获取输入 Shape
    // =========================================================

    auto inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    int64_t totalNum =
        inputShape->GetOriginShape().GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "invalid input shape"),
        return ge::GRAPH_FAILED);


    // =========================================================
    // 4. 获取 dtype
    // =========================================================

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    ge::DataType dataType =
        inputDesc->GetDataType();


    int64_t typeSize = 0;

    if (dataType == ge::DT_FLOAT) {
        typeSize = 4;
    } else if (dataType == ge::DT_FLOAT16) {
        typeSize = 2;
    } else {
        OP_LOGE(
            context,
            "Relu only supports float32 and float16");

        return ge::GRAPH_FAILED;
    }


    // =========================================================
    // 5. 32B 对齐
    //
    // float32 -> 8 elements / block
    // float16 -> 16 elements / block
    // =========================================================

    constexpr int64_t DATA_BLOCK_SIZE = 32;

    int64_t alignNum =
        DATA_BLOCK_SIZE / typeSize;


    // =========================================================
    // 6. Core 切分
    //
    // 输入固定为 8 * 2048 = 16384
    //
    // 选择 32 个 Core：
    //
    // 16384 / 32 = 512
    //
    // 512 对 float32 / float16 都是 32B 对齐
    // =========================================================

    int64_t usedCoreNum = 1;

    if (coreNum >= 32 && totalNum >= 32) {
        usedCoreNum = 32;
    } else {
        usedCoreNum = coreNum;

        // 保证每个 Core 的数据量 32B 对齐
        while (usedCoreNum > 1 &&
               (totalNum / usedCoreNum) % alignNum != 0) {
            --usedCoreNum;
        }
    }

    int64_t blockFactor =
        CeilDiv(totalNum, usedCoreNum);


    // =========================================================
    // 7. UB Tile
    //
    // 使用 Double Buffer
    //
    // Input:
    //   2 * ubFactor * sizeof(T)
    //
    // Output:
    //   2 * ubFactor * sizeof(T)
    //
    // 总共：
    //   4 * ubFactor * sizeof(T)
    // =========================================================

    int64_t totalBufferNum =
        BUFFER_NUM * 2;

    int64_t maxUbElements =
        static_cast<int64_t>(ubSize) /
        (totalBufferNum * typeSize);

    int64_t ubBlockSize =
        GetUbBlockSize(context);

    OP_CHECK_IF(
        ubBlockSize <= 0,
        OP_LOGE(context, "ubBlockSize is invalid"),
        return ge::GRAPH_FAILED);

    int64_t ubAlignElements =
        ubBlockSize / typeSize;

    if (ubAlignElements < alignNum) {
        ubAlignElements = alignNum;
    }


    int64_t ubFactor =
        FloorDiv(
            maxUbElements,
            ubAlignElements) *
        ubAlignElements;


    // Tile 不超过一个 Core 的工作量
    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }


    // =========================================================
    // 8. 安全检查
    // =========================================================

    if (ubFactor <= 0) {
        OP_LOGE(context, "ubFactor is invalid");
        return ge::GRAPH_FAILED;
    }


    // =========================================================
    // 9. 写入 TilingData
    // =========================================================

    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;


    // =========================================================
    // 10. 设置 BlockDim
    // =========================================================

    context->SetBlockDim(
        static_cast<uint32_t>(usedCoreNum));


    // =========================================================
    // 11. 设置 Tiling Key
    // =========================================================

    uint64_t tilingKey;

    if (dataType == ge::DT_FLOAT16) {
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