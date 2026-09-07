/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/relu_tiling_data.h"
// 注意：本 host 文件刻意不再包含 relu_tiling_key.h，以免其条件包含在 host 端
// 意外拉入 kernel_operator.h（设备端头，host 编译找不到）。tilingKey 直接取与
// device 端 relu.cpp 中 ReluTilingKey 枚举一致的字面值：0=fp16/bf16, 1=float。

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
// 与 op_kernel/relu.h 中保持一致：双缓冲队列缓冲数
constexpr int32_t BUFFER_NUM = 2;

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

    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);

    // 计算总元素数量：GetInputShape(0) 返回 const gert::StorageShape*，
    // 其本身无 GetDimNum/GetDim，需先 GetShape() 取出内层 const gert::Shape& 再逐维相乘
    // （CompileTimeTensorDesc 在 cann-9.0.0 无 GetStorageShape 成员，故改用 GetInputShape）
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = inputShapePtr->GetShape();
    int64_t totalNum = 1;
    size_t dimNum = inputShape.GetDimNum();
    if (dimNum == 0) {
        totalNum = 1;  // 标量退化为 1 个元素
    } else {
        for (size_t i = 0; i < dimNum; i++) {
            totalNum *= inputShape.GetDim(i);
        }
    }

    // 数据类型字节数
    int64_t elemSize =
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) ? 2 : 4;

    // 单次搬运需满足 32 字节对齐，换算为元素对齐数
    int64_t elemAlign = 32 / elemSize;

    // UB 单 buffer 可容纳的最大元素数（双缓冲：输入/输出各 BUFFER_NUM 份）
    int64_t maxUbFactor = static_cast<int64_t>(ubSize) / (elemSize * BUFFER_NUM * 2);
    maxUbFactor = (maxUbFactor / elemAlign) * elemAlign;
    if (maxUbFactor <= 0) {
        maxUbFactor = elemAlign;
    }

    // 多核切分 + UB 块大小（性能关键）：
    // 把 ubFactor 用满 UB 容量上限(maxUbFactor)，从而把 tile 总数压到最少，
    // 直接削减每个 tile 的固定开销(AllocTensor/EnQue/DeQue/FreeTensor/循环等)——
    // 这是把最优用时压到 4us 以下的关键（原切分用满 20 核导致 tile 过多、开销累积超 4us）。
    // 每个核只处理 1 个 tile：ReLU 计算极轻，双缓冲重叠收益有限，
    // 优先用“大 tile + 多核”拉满搬运带宽。
    int64_t ubFactor = maxUbFactor;
    int64_t blockFactor = ubFactor;  // 每核 1 个 tile
    int64_t usedCore = CeilDiv(totalNum, blockFactor);
    if (usedCore < 1) {
        usedCore = 1;
    }

    if (usedCore > coreNum) {
        // 海量元素：用满所有核，每核 1 tile，并对齐到 32B
        usedCore = coreNum;
        blockFactor = CeilDiv(totalNum, usedCore);
        blockFactor = (blockFactor / elemAlign) * elemAlign;
        if (blockFactor <= 0) {
            blockFactor = elemAlign;
        }
        ubFactor = blockFactor;
    } else if (blockFactor > totalNum) {
        // 元素总量很小（一个 tile 都放不下整段）：单核整段处理，对齐到 32B
        blockFactor = totalNum;
        ubFactor = (totalNum / elemAlign) * elemAlign;
        if (ubFactor <= 0) {
            ubFactor = elemAlign;
        }
        usedCore = 1;
    }

    // TODO: 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(usedCore));

    // 根据输入 dtype 选择 tilingKey，值与 device 端 relu.cpp 的 ReluTilingKey 枚举一致：
    //   fp16 / bf16 -> 0 (对应 Relu<half>)；float -> 1 (对应 Relu<float>)
    uint64_t tilingKey =
        (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16) ? 0 : 1;
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
