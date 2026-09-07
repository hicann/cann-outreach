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

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr uint32_t BUFFER_NUM = 2;    // 与 kernel 端 BUFFER_NUM 保持一致（DoubleBuffer）
constexpr int64_t UB_RESERVED = 1024; // UB 预留系统开销
constexpr int64_t ALIGN_SIZE = 32;    // 搬运块字节对齐（float16 16 个元素 / float32 8 个元素）

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

    // 输入 shape 逐维相乘得到总元素数（CANN 9.0：GetInputShape 返回 StorageShape）
    const gert::StorageShape* xShape = context->GetInputShape(0);
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= xShape->GetStorageShape().GetDim(i);
    }
    tiling->totalNum = totalNum;

    // 核数选择：优先使用与 totalNum 整除的核心数（向下取不超过平台核数的最大
    // 整除因子），使每个核处理相同元素数、无尾核边界；数据量小于核数时用
    // totalNum 个核（每核 1 个元素）
    if (coreNum > totalNum && totalNum > 0) {
        coreNum = totalNum;
    }
    while (coreNum > 1 && totalNum % coreNum != 0) {
        coreNum--;
    }
    int64_t blockFactor = totalNum / coreNum;

    // 根据输入 dtype 选择 tilingKey（float16/BF16 -> MODE_0 的 half 实例，float32 -> MODE_1 的 float 实例）
    auto inputDesc = context->GetInputDesc(0);
    uint64_t tilingKey = (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) ? 0 : 1;
    context->SetTilingKey(tilingKey);

    // 输入元素字节大小：float16 为 2 字节，其余（float32）为 4 字节
    int64_t byteSize = (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) ? 2 : TYPE_SIZE;

    // UB 空间规划：DoubleBuffer 下输入、输出队列各需 BUFFER_NUM 份缓存，
    // 单份缓存可用字节数 = ubSize / 2 / BUFFER_NUM 并预留少量系统开销，
    // 按 32 字节对齐后换算成每次搬运的元素数
    int64_t perBufferByte = (int64_t)ubSize / 2 / BUFFER_NUM - UB_RESERVED;
    int64_t ubFactor = perBufferByte / ALIGN_SIZE * ALIGN_SIZE / byteSize;
    if (ubFactor <= 0) {
        ubFactor = 8; // 兜底：至少一个 32 字节块
    }
    // 每核数据量不大时一次搬完（kernel 端循环 1 次即可）
    if (blockFactor <= ubFactor) {
        ubFactor = blockFactor;
    }

    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 配置启动核数（多核并行）
    context->SetBlockDim(coreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
