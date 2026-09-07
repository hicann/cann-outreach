/*!
 * \file square_tiling.cpp
 * \brief Square 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t TYPE_SIZE = 4;

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

// 每个元素在 UB 中占用的 buffer 份数：inputQueueX / outputQueueY 各 1 份，
// 且 kernel 侧使用了 BUFFER_NUM = 2 的双缓冲，所以一个元素在 UB 中总共
// 需要 2(in/out) * 2(double buffer) = 4 份同 dtype 大小的空间。
constexpr int64_t UB_QUEUE_NUM = 2;
constexpr int64_t UB_DOUBLE_BUFFER = 2;
// 给系统占用/对齐预留一小部分 UB 空间，避免边界计算误差导致越界
constexpr uint64_t UB_RESERVED_BYTES = 1024;
// 内存访问优化：单次 DataCopy 尽量不小于 16KB，才能较好地打满搬运带宽。
// 用它来决定"值得使用多少个核"——核数太多会把每个核的数据切得过碎，
// 单次搬运长度掉到 16KB 以下反而得不偿失。
constexpr int64_t TARGET_BYTES_PER_CORE = 16 * 1024;

static ge::graphStatus SquareTilingFunc(gert::TilingContext* context)
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

    SquareTilingData* tiling = context->GetTilingData<SquareTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 根据输入 dtype 选择 tilingKey / 每元素字节数
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    ge::DataType dtype = inputDesc->GetDataType();
    bool isFp16 = (dtype == ge::DT_FLOAT16);
    int64_t dtypeSize = isFp16 ? 2 : static_cast<int64_t>(TYPE_SIZE);

    // 计算输入总元素个数：支持任意多维张量 (..., N)，展平为一维处理。
    // Square 是纯逐元素运算，元素之间没有任何依赖，因此不必受限于某一维
    // （例如末维 N）的边界，可以在整个 totalNum 范围内自由切分，
    // 这比"只能按整行/整块切分"的方案能做到更均匀的负载分配。
    const gert::StorageShape* inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inShape = EnsureNotScalar(inputShapePtr->GetStorageShape());
    int64_t totalNum = 1;
    for (size_t i = 0; i < inShape.GetDimNum(); ++i) {
        totalNum *= inShape.GetDim(i);
    }
    if (totalNum < 0) {
        totalNum = 0;
    }

    // ---- 核间切分（Tiling 策略优化：大核 / 小核均匀分配）----
    // 1) 先根据"每个核至少分到一次 ~16KB 大小的搬运"这一目标，估算值得使用的核数，
    //    避免核数过多导致每个核的数据量过碎、单次 DataCopy 效率下降（内存访问优化）；
    // 2) 再和物理核数、数据量本身取三者最小值，得到实际使用的核数 useCoreNum；
    // 3) 用 totalNum / useCoreNum 得到"小核"基准块 avgBlock，
    //    余数 bigCoreNum = totalNum % useCoreNum 个"大核"（排在最前面）
    //    每个核比基准块多处理 1 个元素。
    //    这样任意两个核之间的负载差不超过 1 个元素——是理论上最均匀的整数切分方式，
    //    从根本上避免"余数全部堆到最后一个尾核，其余核提前算完、白白空闲等待"的问题。
    int64_t useCoreNum = 1;
    if (totalNum > 0) {
        int64_t targetElemPerCore = TARGET_BYTES_PER_CORE / dtypeSize;
        if (targetElemPerCore <= 0) {
            targetElemPerCore = 1;
        }
        int64_t coreNumByBandwidth = totalNum / targetElemPerCore;
        if (coreNumByBandwidth < 1) {
            coreNumByBandwidth = 1;
        }
        useCoreNum = coreNum;
        if (useCoreNum > coreNumByBandwidth) {
            useCoreNum = coreNumByBandwidth;
        }
        if (useCoreNum > totalNum) {
            useCoreNum = totalNum; // 核数不能超过元素数，否则会有核分不到数据
        }
        if (useCoreNum < 1) {
            useCoreNum = 1;
        }
    }

    int64_t avgBlock = (totalNum > 0) ? (totalNum / useCoreNum) : 0; // 小核基准块
    int64_t bigCoreNum = (totalNum > 0) ? (totalNum % useCoreNum) : 0; // 大核个数（每个多处理 1 个元素）
    int64_t maxCoreLen = avgBlock + (bigCoreNum > 0 ? 1 : 0); // 单个核最多处理的元素个数，用于限定 ubFactor

    // ---- UB 内切分：单次搬入/计算/搬出的元素个数（内存访问优化：尽量搬运大块数据）----
    uint64_t ubBlockSize = GetUbBlockSize(context); // 硬件 UB 搬运对齐粒度（字节）
    if (ubBlockSize == 0) {
        ubBlockSize = 32; // 兜底值，Ascend 上常见的对齐粒度
    }
    int64_t elemPerBlock = static_cast<int64_t>(ubBlockSize) / dtypeSize;
    if (elemPerBlock <= 0) {
        elemPerBlock = 1;
    }

    uint64_t usableUb = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
    // 双缓冲（BUFFER_NUM=2）是流水编排优化的关键：CopyIn(MTE2)/Compute(V)/CopyOut(MTE3)
    // 使用独立的指令队列，只要 UB 中同时存在两份 buffer，相邻迭代之间这三级流水就能
    // 自动互相掩盖（本次 Compute 与下一次 CopyIn 并行，本次 CopyOut 与下下次 Compute 并行）。
    // 这里没有进一步加深到 3 级、4 级缓冲，是因为每多一级缓冲，单块可用的 UB 空间就减半，
    // 单次搬运长度更容易跌破 16KB 的带宽拐点——双缓冲是"流水掩盖"和"大块搬运"两者的最佳折中。
    int64_t bytesPerElem = UB_QUEUE_NUM * UB_DOUBLE_BUFFER * dtypeSize;
    int64_t ubFactor = static_cast<int64_t>(usableUb) / bytesPerElem;
    // 按硬件对齐粒度向下取整，提升单次 DataCopy/Vector 计算效率
    ubFactor = FloorAlign(ubFactor, elemPerBlock);
    if (ubFactor <= 0) {
        ubFactor = elemPerBlock;
    }
    if (maxCoreLen > 0 && ubFactor > maxCoreLen) {
        // 单个核总共需要处理的数据都不足一个 UB tile 时，没必要多分配 UB 空间
        ubFactor = maxCoreLen;
    }
    if (ubFactor <= 0) {
        ubFactor = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = avgBlock;
    tiling->ubFactor = ubFactor;
    tiling->bigCoreNum = bigCoreNum;

    context->SetBlockDim(useCoreNum > 0 ? useCoreNum : 1);

    uint64_t tilingKey =
        isFp16 ? GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_0) : GET_TPL_TILING_KEY(SQUARE_TPL_SCH_MODE_1);
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSquare([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SquareCompileInfo {};

IMPL_OP_OPTILING(Square).Tiling(SquareTilingFunc).TilingParse<SquareCompileInfo>(TilingParseForSquare);

} // namespace optiling
