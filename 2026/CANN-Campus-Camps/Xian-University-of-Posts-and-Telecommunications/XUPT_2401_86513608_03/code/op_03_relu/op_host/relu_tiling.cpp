#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
namespace optiling {
using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int32_t BUFFER_NUM = 2;

/**
 * @brief 获取平台硬件信息：UB大小、AIV核数量
 * @param context tiling上下文
 * @param ubSize 返回UB大小
 * @param coreNum 返回可用AIV核数
 * @return 执行状态
 */
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

/**
 * @brief 设置workspace，Relu算子不需要额外临时内存
 * @param context tiling上下文
 * @return 执行状态
 */
static ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief Relu算子tiling主函数，完成多核切分、UB块大小计算
 * @param context tiling上下文
 * @return 执行状态
 */
static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    uint64_t ubSize;
    int64_t coreNum;
    // 获取硬件参数
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
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // 读取输入shape，计算总元素个数
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetShape();
    int64_t totalNum = 1;
    size_t dimNum = inputShape.GetDimNum();
    if (dimNum == 0) {
        totalNum = 1;
    } else {
        for (size_t i = 0; i < dimNum; i++) {
            totalNum *= inputShape.GetDim(i);
        }
    }

    // 根据数据类型获取单元素字节，计算32B对齐的元素数
    int64_t elemSize =
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) ? 2 : 4;
    int64_t elemAlign = 32 / elemSize;

    // 根据UB总容量，计算单次tile最大可处理元素，双缓冲输入输出各占BUFFER_NUM份
    int64_t maxUbFactor = static_cast<int64_t>(ubSize) / (elemSize * BUFFER_NUM * 2);
    maxUbFactor = (maxUbFactor / elemAlign) * elemAlign;
    if (maxUbFactor <= 0) {
        maxUbFactor = elemAlign;
    }

    int64_t ubFactor = maxUbFactor;
    int64_t blockFactor = ubFactor;
    // 计算需要使用的核数
    int64_t usedCore = CeilDiv(totalNum, blockFactor);
    if (usedCore < 1) {
        usedCore = 1;
    }
    // 超过硬件最大核数：使用全部核，重新计算每核处理块大小
    if (usedCore > coreNum) {
        usedCore = coreNum;
        blockFactor = CeilDiv(totalNum, usedCore);
        blockFactor = (blockFactor / elemAlign) * elemAlign;
        if (blockFactor <= 0) {
            blockFactor = elemAlign;
        }
        ubFactor = blockFactor;
    // 总数据量很小，单核完成全部计算
    } else if (blockFactor > totalNum) {
        blockFactor = totalNum;
        ubFactor = (totalNum / elemAlign) * elemAlign;
        if (ubFactor <= 0) {
            ubFactor = elemAlign;
        }
        usedCore = 1;
    }

    // 填充tiling参数，下发给device核函数
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    context->SetBlockDim(static_cast<uint32_t>(usedCore));

    // 设置tilingKey，区分fp16/bf16(0)、float(1)，用于选择设备端模板实例
    uint64_t tilingKey =
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) ? 0 : 1;
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

/**
 * @brief tiling解析回调，Relu无需特殊解析逻辑
 */
static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};
// 注册tiling入口
IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);
} // namespace optiling
