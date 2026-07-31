/*!
 * \file truncate_div_tiling.cpp
 * \brief TruncateDiv 算子 Tiling 实现
 *
 * Tiling 策略：
 * 1. 通过 GetInputShape 获取输入 x1, x2 的形状，计算广播形状和总元素数
 * 2. 根据总元素数将工作分配到多个核上
 * 3. 根据数据类型和 UB 大小计算每次迭代处理的元素数
 * 4. 通过 TilingKey 区分 fp16 / fp32 / bf16 三种计算模式（bf16 使用 fp32 中间计算）
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
// Trunc API 临时缓冲区大小直接按公式估算（host 侧无法使用 GetTruncMaxMinTmpSize）
// Trunc<T> 内部做 Cast<float> 转换，需要 count*sizeof(float) 字节中间存储
#include "../op_kernel/truncate_div_tiling_data.h"
#include "../op_kernel/truncate_div_tiling_key.h"

namespace optiling {

using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::Base::FloorDiv;
using Ops::Base::FloorAlign;
using Ops::Base::GetUbBlockSize;

constexpr uint32_t WS_SYS_SIZE = 0U;
constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
constexpr int64_t MIN_ELEMS_PER_CORE = 2048;  // 多核模式下每核最少处理元素数
constexpr int64_t UB_RESERVE_SIZE = 4096;   // UB 保留空间（字节）
constexpr int64_t ALIGN_SIZE = 32;          // 向量化对齐大小

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

/**
 * @brief 根据数据类型和 UB 大小计算单次 UB 循环处理的元素数
 *        同时计算 Trunc 高阶 API 所需的临时缓冲区大小
 */
static int64_t CalculateUbFactor(ge::DataType dtype, uint64_t ubSize, int64_t& tmpSize)
{
    // 使用 TQue 双缓冲: x1(2) + x2(2) + y(2) = 6 slots
    constexpr int64_t DATA_SLOTS = 6;

    int64_t typeSize = 4;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        typeSize = 2;
    }

    int64_t usableUb = static_cast<int64_t>(ubSize) - UB_RESERVE_SIZE;
    if (usableUb <= 0) {
        usableUb = static_cast<int64_t>(ubSize) / 2;
    }

    // bf16 和 fp16 都使用 fp32 中间计算，UB 需求一致：
    // TQue (6 * 2 = 12 bytes/elm) + fp32 TBuf (3 * 4 = 12 bytes/elm) = 24 bytes/elm
    bool isCastPath = (dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT16);

    // 第一轮：忽略 tmpSize 计算 ubFactor
    int64_t ubFactor;
    if (isCastPath) {
        ubFactor = usableUb / 24;
    } else {
        ubFactor = usableUb / (DATA_SLOTS * typeSize);
    }
    ubFactor = FloorAlign(ubFactor, ALIGN_SIZE);
    if (ubFactor < ALIGN_SIZE) {
        ubFactor = ALIGN_SIZE;
    }

    // 根据 ubFactor 计算 Trunc 所需的临时缓冲区大小
    // Trunc<T> 内部做 Cast<float> 转换，需要 ubFactor*sizeof(float) 字节中间存储 + 256B overhead
    {
        tmpSize = ubFactor * static_cast<int64_t>(sizeof(float)) + 256;
    }

    // 第二轮：扣除 tmpSize 后重新计算 ubFactor
    int64_t remainingForData = usableUb - tmpSize;
    if (remainingForData > 0) {
        if (isCastPath) {
            ubFactor = remainingForData / 24;
        } else {
            ubFactor = remainingForData / (DATA_SLOTS * typeSize);
        }
        ubFactor = FloorAlign(ubFactor, ALIGN_SIZE);
        if (ubFactor < ALIGN_SIZE) {
            ubFactor = ALIGN_SIZE;
        }
        // 重新计算 tmpSize（ubFactor 可能已经变化）
        tmpSize = ubFactor * static_cast<int64_t>(sizeof(float)) + 256;
    }

    return ubFactor;
}

static ge::graphStatus TruncateDivTilingFunc(gert::TilingContext* context)
{
    // 1. 获取平台信息
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

    // 2. 获取输入 shape（TilingContext 使用 GetInputShape 返回 StorageShape）
    auto* x1ShapePtr = context->GetInputShape(0);
    auto* x2ShapePtr = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1ShapePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context, x2ShapePtr);

    // 3. 获取数据类型
    auto inputDesc0 = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc0);
    ge::DataType dtype = inputDesc0->GetDataType();

    // 4. 广播形状计算：提取维度，靠右对齐，逐维广播
    //    标量 (GetDimNum==0) 视为 shape={1}，可广播到任意维度
    const gert::Shape& x1Shape = x1ShapePtr->GetShape();
    const gert::Shape& x2Shape = x2ShapePtr->GetShape();
    size_t origDimNum1 = x1Shape.GetDimNum();
    size_t origDimNum2 = x2Shape.GetDimNum();

    size_t effDimNum1 = (origDimNum1 == 0) ? 1 : origDimNum1;
    size_t effDimNum2 = (origDimNum2 == 0) ? 1 : origDimNum2;
    size_t maxDimNum = (effDimNum1 > effDimNum2) ? effDimNum1 : effDimNum2;

    if (maxDimNum > kMaxDim) {
        OP_LOGE(context, "dimension count %zu exceeds kMaxDim %lld", maxDimNum, kMaxDim);
        return ge::GRAPH_FAILED;
    }

    int64_t outShape[kMaxDim] = {};
    int64_t x1ShapeArr[kMaxDim] = {};
    int64_t x2ShapeArr[kMaxDim] = {};
    int64_t totalNum = 1;
    int64_t x1ElemNum = 1;
    int64_t x2ElemNum = 1;

    for (size_t i = 0; i < maxDimNum; i++) {
        int64_t dim1 = 1;
        int64_t dim2 = 1;

        // x1: 非标量时才读取实际维度
        if (origDimNum1 > 0 && i >= maxDimNum - effDimNum1) {
            dim1 = x1Shape.GetDim(i - (maxDimNum - effDimNum1));
        }
        // x2: 非标量时才读取实际维度
        if (origDimNum2 > 0 && i >= maxDimNum - effDimNum2) {
            dim2 = x2Shape.GetDim(i - (maxDimNum - effDimNum2));
        }

        x1ShapeArr[i] = dim1;
        x2ShapeArr[i] = dim2;
        x1ElemNum *= dim1;
        x2ElemNum *= dim2;

        if (dim1 == dim2) {
            outShape[i] = dim1;
        } else if (dim1 == 1) {
            outShape[i] = dim2;
        } else if (dim2 == 1) {
            outShape[i] = dim1;
        } else {
            OP_LOGE(context, "Broadcast shape incompatible: dim1=%ld, dim2=%ld", dim1, dim2);
            return ge::GRAPH_FAILED;
        }
        totalNum *= outShape[i];
    }

    // 5. 分核策略：小 shape 单核，大 shape 按 MIN_ELEMS_PER_CORE 控制核数避免过度拆分
    int64_t blockDim;
    int64_t perCoreNum;

    if (totalNum <= MIN_SPLIT_THRESHOLD) {
        blockDim = 1;
        perCoreNum = totalNum;
    } else {
        // 限制最大核数：确保每核至少处理 MIN_ELEMS_PER_CORE 个元素
        int64_t maxCores = std::max(int64_t(1), CeilDiv(totalNum, MIN_ELEMS_PER_CORE));
        blockDim = std::min(coreNum, maxCores);
        perCoreNum = CeilDiv(totalNum, blockDim);
    }

    // 多核场景下对齐 perCoreNum 到 ALIGN_SIZE 边界，确保各核 GM 偏移满足 32 字节对齐
    // 单核模式跳过对齐（Core 0 从 offset=0 开始，天然对齐）
    // 使用 CeilAlign（向上对齐），避免 FloorAlign 导致 perCoreNum*blockDim < totalNum 丢失尾部元素
    if (blockDim > 1) {
        int64_t alignTypeSize = (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) ? 2 : 4;
        int64_t alignElems = std::max(int64_t(1), ALIGN_SIZE / alignTypeSize);
        perCoreNum = CeilAlign(perCoreNum, alignElems);
    }

    // 6. 计算 UB 单次循环处理的元素数（同时获取 Trunc 临时缓冲区大小）
    int64_t tmpSize = 0;
    int64_t ubFactor = CalculateUbFactor(dtype, ubSize, tmpSize);

    if (ubFactor > perCoreNum) {
        ubFactor = FloorAlign(perCoreNum, ALIGN_SIZE);
        if (ubFactor < ALIGN_SIZE) {
            ubFactor = perCoreNum;
        }
    }
    if (ubFactor > totalNum) {
        ubFactor = totalNum;
    }

    // 7. 设置 tiling 数据
    TruncateDivTilingData* tiling = context->GetTilingData<TruncateDivTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    tiling->totalNum = totalNum;
    tiling->blockFactor = perCoreNum;
    tiling->ubFactor = ubFactor;
    tiling->x1ElemNum = x1ElemNum;
    tiling->x2ElemNum = x2ElemNum;
    tiling->dimNum = static_cast<int64_t>(maxDimNum);
    tiling->tmpSize = tmpSize;
    for (size_t i = 0; i < maxDimNum; i++) {
        tiling->outShape[i] = outShape[i];
        tiling->x1Shape[i] = x1ShapeArr[i];
        tiling->x2Shape[i] = x2ShapeArr[i];
    }

    // 8. 设置 block dimension
    context->SetBlockDim(blockDim);

    // 9. 根据输入 dtype 选择 tilingKey
    // Mode 0: fp16 (2 字节), Mode 1: fp32 (4 字节), Mode 2: bf16 (2 字节, fp32 中间计算)
    uint64_t tilingKey;
    if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEDIV_TPL_SCH_MODE_2);
    } else if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEDIV_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEDIV_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForTruncateDiv([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct TruncateDivCompileInfo {};

IMPL_OP_OPTILING(TruncateDiv).Tiling(TruncateDivTilingFunc).TilingParse<TruncateDivCompileInfo>(TilingParseForTruncateDiv);

} // namespace optiling
