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

    // 1. 输入为必选 tensor：直接取其元素总数与 dtype（与框架参考写法一致）
    const gert::Tensor* inputTensor = context->GetRequiredInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputTensor);
    int64_t totalNum = static_cast<int64_t>(inputTensor->GetShapeSize());
    ge::DataType dtype = inputTensor->GetDataType();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "invalid input size"), return ge::GRAPH_FAILED);

    // 2. 根据 dtype 选择 tilingKey 与元素字节数
    //    tilingKey 0 -> kernel 实例 Relu<half>（fp16）
    //    tilingKey 1 -> kernel 实例 Relu<float>（fp32）
    int64_t typeSize = 4;
    uint64_t tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    if (dtype == ge::DT_FLOAT16) {
        typeSize = 2;
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    }

    // 3. 元素按 32B 对齐后切分，保证各核起始地址与 DataCopy 长度都对齐
    constexpr int64_t ALIGN_BYTES = 32;
    const int64_t alignElems = ALIGN_BYTES / typeSize;  // fp32: 8, fp16: 16
    OP_CHECK_IF(
        totalNum % alignElems != 0,
        OP_LOGE(context, "totalNum must be aligned to 32Bytes"),
        return ge::GRAPH_FAILED);
    const int64_t totalUnits = totalNum / alignElems;

    // 4. 核数：本算例数据量小，核过多只会放大调度与收尾开销；目标 8 核即可。
    //    取 <= min(TARGET_CORES, 平台AIV核数, 总块数) 的最大整除因子，保证各核
    //    负载相同、起始地址与 DataCopy 长度都对齐、不越界。
    constexpr int64_t TARGET_CORES = 8;
    int64_t maxCores = totalUnits < TARGET_CORES ? totalUnits : TARGET_CORES;
    if (coreNum < maxCores) {
        maxCores = coreNum;
    }
    int64_t usedCores = 1;
    for (int64_t c = maxCores; c >= 1; --c) {
        if (totalUnits % c == 0) {
            usedCores = c;
            break;
        }
    }
    const int64_t blockFactor = (totalUnits / usedCores) * alignElems;

    // 5. 单 tile 元素数：in/out 双缓冲(DoubleBuffer)预留，最多占 ubSize/4
    //    同时保证是 alignElems 的整数倍
    int64_t perTileUnits = (int64_t)(ubSize / 4U) / (typeSize * alignElems);
    if (perTileUnits < 1) {
        perTileUnits = 1;
    }
    if (perTileUnits > totalUnits / usedCores) {
        perTileUnits = totalUnits / usedCores;
    }
    const int64_t ubFactor = perTileUnits * alignElems;

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<int32_t>(usedCores));
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
