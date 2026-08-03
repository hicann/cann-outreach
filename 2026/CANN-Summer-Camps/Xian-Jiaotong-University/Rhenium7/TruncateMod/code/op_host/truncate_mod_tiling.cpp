/*!
 * \file truncate_mod_tiling.cpp
 * \brief TruncateMod 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/truncate_mod_tiling_data.h"
#include "../op_kernel/truncate_mod_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t EXTRA_UB_SIZE = 1024;   // UB 预留（含 scalarBuf 等），DESIGN.md §1.5
constexpr int64_t MIN_BYTES_PER_CORE = 4096; // 每核至少 4KB 数据，DESIGN.md §2.1
constexpr int64_t BLOCK_ALIGN_ELEM = 512;    // 多核切分 512 元素对齐，DESIGN.md §2.1
constexpr int64_t BUFFER_DIVISOR_HALF = 24;  // mode0/1（T=2B）UB 每元素字节数，DESIGN.md §1.5
constexpr int64_t BUFFER_DIVISOR_FLOAT = 28; // mode2（T=4B）UB 每元素字节数，DESIGN.md §1.5

// 最大不超过 cap 的 n 的因数（n <= cap 时取 n 本身；否则向下扫描）
static inline int64_t LargestDivisorLE(int64_t n, int64_t cap)
{
    if (n <= cap) {
        return n;
    }
    for (int64_t d = cap; d >= 1; d--) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

static inline int64_t Gcd(int64_t a, int64_t b)
{
    while (b != 0) {
        int64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static inline int64_t Lcm(int64_t a, int64_t b)
{
    if (a == 0 || b == 0) {
        return 0;
    }
    return a / Gcd(a, b) * b;
}

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

// 计算输入总元素数（rank-0 标量按 1 处理，DESIGN.md §3.1 EnsureNotScalar 语义）
static inline int64_t GetShapeTotal(const gert::Shape& shape)
{
    int64_t total = 1;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        total *= shape.GetDim(i);
    }
    return total;
}

// row-run 描述（DESIGN.md §2.4.3）：
// 输入按输出 rank 右对齐后，从最内维向左扫描，与输出相同的连续段累乘为 innerLen，
// 首个广播轴（输入 dim=1、输出>1）处停止；rowCount = inTotal / innerLen。
// 标量输入退化为 (innerLen=1, rowCount=1)。
static inline void ComputeRowRun(const gert::Shape& inShape, const gert::Shape& outShape,
                                 int64_t& innerLen, int64_t& rowCount)
{
    const int64_t inTotal = GetShapeTotal(inShape);
    const size_t rankIn = inShape.GetDimNum();
    const size_t rankOut = outShape.GetDimNum();

    innerLen = 1;
    for (int d = static_cast<int>(rankOut) - 1; d >= 0; d--) {
        int64_t inDim = 1;
        if (rankIn > 0 && static_cast<size_t>(d) >= rankOut - rankIn) {
            inDim = inShape.GetDim(d - (rankOut - rankIn));
        }
        const int64_t outDim = outShape.GetDim(d);
        if (inDim == outDim) {
            innerLen *= inDim;
        } else {
            break;
        }
    }
    rowCount = inTotal / innerLen;
}

// 广播合法性 + row-run 模型有效性校验（design_issue 防护，PLAN.md §5）：
//  1. 右对齐广播逐维校验（与 infershape 一致的防御性检查，保证 tiling UT 可独立判定非法广播）；
//  2. row-run 模型仅对 rowCount==1 正确：rowCount>1 对应列广播/中间轴广播
//     （如 [64,1]→[64,64]、[2,1,3]→[2,4,3]），kernel 的 (s/innerLen)%rowCount 行映射
//     会产生静默错误结果，必须在此拒绝（numpy 复验见 PLAN.md §5）。
static inline bool BroadcastAndRowRunValid(const gert::Shape& x1Shape, const gert::Shape& x2Shape,
                                           const gert::Shape& yShape,
                                           int64_t& x1InnerLen, int64_t& x1RowCount,
                                           int64_t& x2InnerLen, int64_t& x2RowCount)
{
    const size_t rankX1 = x1Shape.GetDimNum();
    const size_t rankX2 = x2Shape.GetDimNum();
    const size_t rankOut = yShape.GetDimNum();
    const size_t rankMax = (rankX1 > rankX2) ? rankX1 : rankX2;
    if (rankOut != rankMax) {
        return false;
    }
    for (size_t d = 0; d < rankOut; d++) {
        int64_t dimA = 1;
        int64_t dimB = 1;
        if (rankX1 > 0 && d >= rankOut - rankX1) {
            dimA = x1Shape.GetDim(d - (rankOut - rankX1));
        }
        if (rankX2 > 0 && d >= rankOut - rankX2) {
            dimB = x2Shape.GetDim(d - (rankOut - rankX2));
        }
        if (dimA == dimB || dimA == 1 || dimB == 1) {
            continue;
        }
        return false;
    }

    ComputeRowRun(x1Shape, yShape, x1InnerLen, x1RowCount);
    ComputeRowRun(x2Shape, yShape, x2InnerLen, x2RowCount);
    return (x1RowCount == 1) && (x2RowCount == 1);
}

static ge::graphStatus TruncateModTilingFunc(gert::TilingContext* context)
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

    TruncateModTilingData* tiling = context->GetTilingData<TruncateModTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    const gert::StorageShape* x1StorageShape = context->GetInputShape(0);
    const gert::StorageShape* x2StorageShape = context->GetInputShape(1);
    const gert::StorageShape* yStorageShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1StorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2StorageShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorageShape);

    const gert::Shape x1Shape = EnsureNotScalar(x1StorageShape->GetStorageShape());
    const gert::Shape x2Shape = EnsureNotScalar(x2StorageShape->GetStorageShape());
    const gert::Shape yShape = EnsureNotScalar(yStorageShape->GetStorageShape());

    // 根据输入 dtype 选择 tilingKey 与 UB bufferDivisor（x1/x2 dtype 一致性已在 InferShape 校验）
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    uint64_t tilingKey = 0;
    int64_t dtypeSize = 4;
    int64_t bufferDivisor = BUFFER_DIVISOR_FLOAT;
    switch (inputDesc->GetDataType()) {
        case ge::DT_FLOAT16:
            tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_0);
            dtypeSize = 2;
            bufferDivisor = BUFFER_DIVISOR_HALF;
            break;
        case ge::DT_BF16:
            tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_1);
            dtypeSize = 2;
            bufferDivisor = BUFFER_DIVISOR_HALF;
            break;
        case ge::DT_FLOAT:
            tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_2);
            dtypeSize = 4;
            bufferDivisor = BUFFER_DIVISOR_FLOAT;
            break;
        default:
            OP_LOGE(context, "TruncateMod unsupported input dtype: %d", inputDesc->GetDataType());
            return ge::GRAPH_FAILED;
    }

    const int64_t totalNum = GetShapeTotal(yShape);
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum=%ld invalid", totalNum), return ge::GRAPH_FAILED);

    // row-run 描述（DESIGN.md §2.4.3）+ 广播合法性/模型有效性防护（PLAN.md §5）
    int64_t x1InnerLen = 0;
    int64_t x1RowCount = 0;
    int64_t x2InnerLen = 0;
    int64_t x2RowCount = 0;
    if (!BroadcastAndRowRunValid(x1Shape, x2Shape, yShape,
                                 x1InnerLen, x1RowCount, x2InnerLen, x2RowCount)) {
        OP_LOGE(context,
                "TruncateMod broadcast unsupported by row-run model (rowCount>1) or invalid: "
                "x1InnerLen=%ld x1RowCount=%ld x2InnerLen=%ld x2RowCount=%ld",
                x1InnerLen, x1RowCount, x2InnerLen, x2RowCount);
        return ge::GRAPH_FAILED;
    }

    // UB 切分（DESIGN.md §2.2 + DataCopyPad 32B 对齐修正，PLAN.md §5）：
    // DataCopyPad GM→UB 的 dst 必须 32B 对齐，且每个 blockLen 非 32B 对齐的数据块会在
    // UB 内填充至 32B——因此多段拼接（kernel 策略 2/3）仅当周期 32B 对齐且 tile 按周期
    // 对齐时才正确。对「行重复」输入（innerLen < total），本实现统一采用单行 tile
    // （strategy 1 恒成立）：
    //   ubFactor    = gcd(行重复 innerLen) 的最大因数（≤ maxElemNum）
    //   blockFactor = CeilAlign(perCore, lcm(512, 行重复 innerLen))
    // 纯 ELEWISE（innerLen == total）输入无行边界问题，沿用原有逻辑。
    const int64_t maxElemNum = (static_cast<int64_t>(ubSize) - EXTRA_UB_SIZE) / bufferDivisor;
    const int64_t alignElem = 256 / dtypeSize;

    const int64_t x1Period = x1InnerLen * x1RowCount;
    const int64_t x2Period = x2InnerLen * x2RowCount;

    int64_t ubFactor = 0;
    int64_t rowRepGcd = 0;
    int64_t rowRepLcm = 1;
    bool hasRowRep = false;
    if (x1Period > 1 && x1InnerLen < totalNum) { // x1 行重复（标量 x1Period==1 走标量路径）
        rowRepGcd = (rowRepGcd == 0) ? x1InnerLen : Gcd(rowRepGcd, x1InnerLen);
        rowRepLcm = Lcm(rowRepLcm, x1InnerLen);
        hasRowRep = true;
    }
    if (x2Period > 1 && x2InnerLen < totalNum) { // x2 行重复
        rowRepGcd = (rowRepGcd == 0) ? x2InnerLen : Gcd(rowRepGcd, x2InnerLen);
        rowRepLcm = Lcm(rowRepLcm, x2InnerLen);
        hasRowRep = true;
    }

    if (hasRowRep) {
        // 行重复输入：单行 tile 模式
        const int64_t g = (rowRepGcd == 0) ? 1 : rowRepGcd;
        ubFactor = LargestDivisorLE(g, maxElemNum);
        if (ubFactor < 1) {
            ubFactor = 1;
        }
    } else {
        // 纯 ELEWISE/标量：原有 256B repeat 对齐 + period 整数倍优化
        ubFactor = FloorAlign(maxElemNum, alignElem);
        if (ubFactor < 1) {
            ubFactor = 1;
        }
        const int64_t minPeriod = (x1Period < x2Period) ? x1Period : x2Period;
        if (minPeriod > 1 && minPeriod % alignElem == 0 && minPeriod <= maxElemNum) {
            const int64_t periodUb = (maxElemNum / minPeriod) * minPeriod;
            if (periodUb >= alignElem) {
                ubFactor = periodUb;
            }
        }
    }

    // 多核切分（DESIGN.md §2.1）：每核至少 4KB 数据 → coreNum；blockFactor 512 元素对齐。
    // 行重复输入另需 blockFactor 为 innerLen 的倍数（保证每个 block 起始在行边界上）。
    int64_t coreNumByData = CeilDiv(totalNum * dtypeSize, MIN_BYTES_PER_CORE);
    int64_t usedCoreNum = (coreNumByData < coreNum) ? coreNumByData : coreNum;
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }
    const int64_t perCore = CeilDiv(totalNum, usedCoreNum);
    int64_t blockFactor = 0;
    if (hasRowRep) {
        blockFactor = CeilAlign(perCore, Lcm(BLOCK_ALIGN_ELEM, rowRepLcm));
    } else {
        blockFactor = CeilAlign(perCore, BLOCK_ALIGN_ELEM);
    }

    // 填充 tilingData
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;
    tiling->x1InnerLen = x1InnerLen;
    tiling->x1RowCount = x1RowCount;
    tiling->x2InnerLen = x2InnerLen;
    tiling->x2RowCount = x2RowCount;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));
    context->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTruncateMod([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TruncateModCompileInfo {};

IMPL_OP_OPTILING(TruncateMod).Tiling(TruncateModTilingFunc).TilingParse<TruncateModCompileInfo>(TilingParseForTruncateMod);

} // namespace optiling
