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


/*
 * Scalar 输入转换成长度为1的Shape
 */
static inline const gert::Shape EnsureNotScalar(
    const gert::Shape& in_shape)
{
    if (in_shape.GetDimNum() == 0) {
        return g_vec_1_shape;
    }

    return in_shape;
}


/*
 * 获取平台 UB 大小和 Vector Core 数量
 */
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


/*
 * Workspace 配置
 */
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


/*
 * ReLU Tiling
 */
static ge::graphStatus ReluTilingFunc(
    gert::TilingContext* context)
{
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    /*
     * 1. 获取平台信息
     */
    OP_CHECK_IF(
        GetPlatformInfo(
            context,
            ubSize,
            coreNum) != ge::GRAPH_SUCCESS,

        OP_LOGE(
            context,
            "GetPlatformInfo error"),

        return ge::GRAPH_FAILED);


    /*
     * 2. 配置 workspace
     */
    OP_CHECK_IF(
        GetWorkspaceSize(context) !=
            ge::GRAPH_SUCCESS,

        OP_LOGE(
            context,
            "GetWorkspaceSize error"),

        return ge::GRAPH_FAILED);


    /*
     * 3. 获取 TilingData
     */
    ReluTilingData* tiling =
        context->GetTilingData<
            ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        tiling);


    /*
     * 4. 获取真实输入 Shape
     */
    const gert::StorageShape* inputShape =
        context->GetInputShape(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputShape);

    const gert::Shape& storageShape =
        inputShape->GetStorageShape();

    const gert::Shape shape =
        EnsureNotScalar(storageShape);


    /*
     * 5. 计算总元素数量
     *
     * 本题 (8, 2048)：
     * totalNum = 16384
     */
    int64_t totalNum = 1;

    for (size_t i = 0;
         i < shape.GetDimNum();
         ++i)
    {
        totalNum *= shape.GetDim(i);
    }

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "totalNum <= 0"),
        return ge::GRAPH_FAILED);


    /*
     * 6. 为了优先保证正确性，使用单核
     *
     * blockFactor 表示每个核处理的元素数量。
     *
     * 单核时：
     * blockFactor = totalNum
     */
    int64_t blockNum = 1;
    int64_t blockFactor = totalNum;


    /*
     * 7. UB 每次处理的元素数量
     *
     * 本题固定Shape为16384，
     * 选择4096后正好分成4个Tile。
     *
     * FP32:
     * 4096 * 4B = 16KB / buffer
     *
     * FP16:
     * 4096 * 2B = 8KB / buffer
     *
     * 输入输出各Double Buffer，
     * UB空间足够。
     */
    int64_t ubFactor = 4096;

    if (ubFactor > blockFactor) {
        ubFactor = blockFactor;
    }


    /*
     * 8. 写入 TilingData
     */
    tiling->totalNum =
        totalNum;

    tiling->blockFactor =
        blockFactor;

    tiling->ubFactor =
        ubFactor;


    /*
     * 9. 启动核数
     */
    context->SetBlockDim(
        blockNum);


    /*
     * 10. 根据 dtype 选择 Kernel 模板
     *
     * relu.cpp 中：
     *
     * MODE_0 -> half
     * MODE_1 -> float
     */
    uint64_t tilingKey;

    auto inputDesc =
        context->GetInputDesc(0);

    OP_CHECK_NULL_WITH_CONTEXT(
        context,
        inputDesc);

    if (inputDesc->GetDataType() ==
            ge::DT_FLOAT16)
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


/*
 * Tiling Parse
 */
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