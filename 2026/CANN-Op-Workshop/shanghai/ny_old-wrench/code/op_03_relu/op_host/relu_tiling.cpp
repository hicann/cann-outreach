/*!
 * \file relu_tiling.cpp
 * \brief Relu 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include <algorithm>
#include "../op_kernel/relu_tiling_data.h"
#include "../op_kernel/relu_tiling_key.h"

namespace optiling {

// 多核切分：每核最小数据量 4KB（单位 bits），保证开核收益
constexpr uint64_t MIN_TILING_BITS_PER_CORE = 32768;
// 多核切分：每核基础数据量按 512 元素对齐，保证 GM 访问效率与核间偏移整齐
constexpr uint64_t ELEM_ALIGN_FACTOR = 512;
// UB 切分：搬运量按 256B 对齐，保证 Vector 指令效率
constexpr uint64_t UB_ALIGN_BYTES = 256;
// 双缓冲：in/out 各 2 份 buffer，CopyIn/Compute/CopyOut 流水并行
constexpr uint64_t BUFFER_NUM = 2;
// UB 系统预留（框架/调度占用），计算可用 UB 时扣除
constexpr uint64_t UB_SYSTEM_RESERVE = 65536;

static ge::graphStatus ReluTilingFunc(gert::TilingContext* context)
{
    // 输入 dtype 决定 tilingKey（float / float16 两套模板实例）
    auto inputDesc = context->GetInputDesc(0);
    OP_CHECK_IF(inputDesc == nullptr, OP_LOGE("ReluTilingFunc", "input desc is nullptr"),
        return ge::GRAPH_FAILED);
    uint32_t dtX = static_cast<uint32_t>(inputDesc->GetDataType());
    ASCENDC_TPL_SEL_PARAM(context, dtX);

    // 输入 shape 展平为元素总数
    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_IF(xShape == nullptr, OP_LOGE("ReluTilingFunc", "input shape is nullptr"),
        return ge::GRAPH_FAILED);
    int64_t dim0 = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        dim0 *= xShape->GetStorageShape().GetDim(i);
    }
    OP_CHECK_IF(dim0 <= 0, OP_LOGE("ReluTilingFunc", "invalid input shape, dim0=%ld", dim0),
        return ge::GRAPH_FAILED);

    // 平台信息：AIV 核数与 UB 大小
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE("ReluTilingFunc", "platform info is nullptr"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(coreNum == 0 || ubSize == 0, OP_LOGE("ReluTilingFunc", "invalid platform info"),
        return ge::GRAPH_FAILED);

    uint32_t elemBytes = static_cast<uint32_t>(ge::GetSizeByDataType(inputDesc->GetDataType()));
    OP_CHECK_IF(elemBytes == 0, OP_LOGE("ReluTilingFunc", "unsupported dtype"),
        return ge::GRAPH_FAILED);

    // ---------- 多核切分 ----------
    // 每核至少处理 MIN_TILING_BITS_PER_CORE bits 数据，核数上限为实际 AIV 核数
    uint64_t needCoreNum = (static_cast<uint64_t>(dim0) * elemBytes * 8 + MIN_TILING_BITS_PER_CORE - 1) /
                           MIN_TILING_BITS_PER_CORE;
    uint64_t useCoreNum = std::min<uint64_t>(needCoreNum, coreNum);
    // 每核基础处理量：向上取整再按 512 元素对齐
    uint64_t blockFormer = (static_cast<uint64_t>(dim0) + useCoreNum - 1) / useCoreNum;
    blockFormer = (blockFormer + ELEM_ALIGN_FACTOR - 1) / ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
    uint64_t blockNum = (static_cast<uint64_t>(dim0) + blockFormer - 1) / blockFormer;
    uint64_t blockTail = static_cast<uint64_t>(dim0) - (blockNum - 1) * blockFormer;

    // ---------- UB 切分 ----------
    // in/out 队列各 BUFFER_NUM 份 buffer：ubFormer * elemBytes * BUFFER_NUM * 2 <= ubSize - 系统预留
    uint64_t bufferDivisor = BUFFER_NUM * 2 * elemBytes;
    uint64_t maxElemNum = (ubSize - UB_SYSTEM_RESERVE) / bufferDivisor;
    // 对齐因子：256B 对齐对应的元素数（fp32=64, fp16=128）
    uint64_t alignFactor = UB_ALIGN_BYTES / elemBytes;
    uint64_t ubFormer = maxElemNum / alignFactor * alignFactor;
    OP_CHECK_IF(ubFormer == 0, OP_LOGE("ReluTilingFunc", "ub size too small"), return ge::GRAPH_FAILED);

    // 首核（blockFormer）的循环次数与尾块大小；尾核数据量更小，循环次数由 kernel 按 blockTail 推出
    uint64_t ubLoop = (blockFormer + ubFormer - 1) / ubFormer;
    uint64_t ubTail = blockFormer - (ubLoop - 1) * ubFormer;

    // ---------- 写入 tiling data ----------
    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_IF(tiling == nullptr, OP_LOGE("ReluTilingFunc", "tiling data is nullptr"),
        return ge::GRAPH_FAILED);
    tiling->totalLength = static_cast<uint64_t>(dim0);
    tiling->blockFormer = blockFormer;
    tiling->blockTail = blockTail;
    tiling->blockNum = blockNum;
    tiling->ubFormer = ubFormer;
    tiling->ubLoop = ubLoop;
    tiling->ubTail = ubTail;

    // 启动核数
    context->SetBlockDim(static_cast<uint32_t>(blockNum));

    // workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_IF(currentWorkspace == nullptr, OP_LOGE("ReluTilingFunc", "workspace is nullptr"),
        return ge::GRAPH_FAILED);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
