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
constexpr int64_t BUFFER_NUM = 2;
constexpr int64_t QUEUE_NUM = 2;  // 一个输入队列、一个输出队列
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
    uint64_t ubSize = 0;
    int64_t coreNum = 0;

    OP_CHECK_IF(
        GetPlatformInfo(context, ubSize, coreNum) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetPlatformInfo error"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "GetWorkspaceSize error"),
        return ge::GRAPH_FAILED);

    // 1. 获取输入 Shape
    const gert::StorageShape* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);

    const gert::Shape storageShape =
        EnsureNotScalar(inputShape->GetStorageShape());

    const int64_t totalNum = storageShape.GetShapeSize();

    OP_CHECK_IF(
        totalNum <= 0,
        OP_LOGE(context, "Invalid totalNum: %ld", totalNum),
        return ge::GRAPH_FAILED);

    // 2. 获取输入数据类型
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    const ge::DataType inputDtype = inputDesc->GetDataType();

    OP_CHECK_IF(
        inputDtype != ge::DT_FLOAT &&
            inputDtype != ge::DT_FLOAT16,
        OP_LOGE(context, "Unsupported input dtype"),
        return ge::GRAPH_FAILED);

    /*
     * 使用 FP32 的 4 字节作为最坏情况进行 UB 规划。
     *
     * GetUbBlockSize() 通常表示一次数据块的字节数；
     * alignNum 表示一个数据块可以容纳的 FP32 元素数。
     */
    const int64_t ubBlockSize =
    static_cast<int64_t>(GetUbBlockSize(context));

    const int64_t alignNum =
        ubBlockSize / TYPE_SIZE;

    OP_CHECK_IF(
        alignNum <= 0,
        OP_LOGE(context, "Invalid alignNum"),
        return ge::GRAPH_FAILED);

    /*
     * 3. 计算使用核数
     *
     * 每个核至少处理 MIN_SPLIT_THRESHOLD=1024 个元素。
     * 对于 16384 个元素：
     *
     *   usedCoreNum = 16384 / 1024 = 16
     */
    int64_t usedCoreNum =
        CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);

    usedCoreNum =
        usedCoreNum > coreNum ? coreNum : usedCoreNum;

    usedCoreNum =
        usedCoreNum < 1 ? 1 : usedCoreNum;

    /*
     * 4. 计算每核负责的元素数，并按数据块对齐
     */
    int64_t blockFactor =
        CeilAlign(CeilDiv(totalNum, usedCoreNum), alignNum);

    // 对齐可能改变 blockFactor，因此重新计算实际核数
    usedCoreNum = CeilDiv(totalNum, blockFactor);

    /*
     * 5. 计算 UB 最大可容纳的元素数
     *
     * UB 中有：
     *   2 个队列：输入、输出
     *   每个队列 2 个 Buffer
     *   每个元素最多占 4 字节
     *
     * UB 使用量：
     *   ubFactor × TYPE_SIZE × QUEUE_NUM × BUFFER_NUM
     */
    const int64_t maxUbFactor = FloorAlign(
        FloorDiv(
            static_cast<int64_t>(ubSize),
            TYPE_SIZE * QUEUE_NUM * BUFFER_NUM),
        alignNum);

    OP_CHECK_IF(
        maxUbFactor <= 0,
        OP_LOGE(context, "UB is too small"),
        return ge::GRAPH_FAILED);

    /*
     * 将每核数据至少切为两块，使 Double Buffer 真正产生作用。
     *
     * 固定输入下：
     *   blockFactor = 1024
     *   ubFactor    = 512
     */
    int64_t ubFactor =
        CeilAlign(CeilDiv(blockFactor, BUFFER_NUM), alignNum);

    ubFactor =
        ubFactor > maxUbFactor ? maxUbFactor : ubFactor;

    OP_CHECK_IF(
        ubFactor <= 0,
        OP_LOGE(context, "Invalid ubFactor"),
        return ge::GRAPH_FAILED);

    // 6. 写入 TilingData
    ReluTilingData* tiling =
        context->GetTilingData<ReluTilingData>();

    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 7. 设置 Kernel 启动核数
    context->SetBlockDim(
        static_cast<uint32_t>(usedCoreNum));

    // 8. 根据 dtype 设置 TilingKey
    uint64_t tilingKey = 0;

    if (inputDtype == ge::DT_FLOAT16) {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey =
            GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
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
