/*
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

    if (coreNum == 0) {
        OP_LOGE(context, "coreNum is 0");
        return ge::GRAPH_FAILED;
    }

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0) {
        OP_LOGE(context, "ubSize is 0");
        return ge::GRAPH_FAILED;
    }

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

    // 获取硬件平台信息
    ge::graphStatus ret = GetPlatformInfo(context, ubSize, coreNum);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "GetPlatformInfo error");
        return ge::GRAPH_FAILED;
    }

    ret = GetWorkspaceSize(context);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "GetWorkspaceSize error");
        return ge::GRAPH_FAILED;
    }

    ReluTilingData* tiling = context->GetTilingData<ReluTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

    // 获取输入张量总元素个数
    const auto* inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape storageShape = EnsureNotScalar(inputShape->GetStorageShape());
    const int64_t totalNum = storageShape.GetShapeSize();

    if (totalNum <= 0) {
        OP_LOGE(context, "invalid totalNum");
        return ge::GRAPH_FAILED;
    }

    // 获取输入数据类型
    const auto* inputDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    const ge::DataType dataType = inputDesc->GetDataType();

    int64_t typeSize = TYPE_SIZE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        typeSize = 2;
    }

    // 32字节对齐参数
    constexpr int64_t BLOCK_BYTES = 32;
    int64_t alignNum = BLOCK_BYTES / typeSize;

    // 初步计算需要开启多少核
    int64_t usedCoreNum = CeilDiv(totalNum, MIN_SPLIT_THRESHOLD);
    if (usedCoreNum > coreNum) {
        usedCoreNum = coreNum;
    }
    if (usedCoreNum < 1) {
        usedCoreNum = 1;
    }

    // 慢慢往下减少核心数，寻找可以整除并且对齐的核数
    bool findFlag = false;
    while (usedCoreNum > 1) {
        int64_t perCoreElem = totalNum / usedCoreNum;
        if ((totalNum % usedCoreNum == 0) && (perCoreElem % alignNum == 0)) {
            findFlag = true;
            break;
        }
        usedCoreNum = usedCoreNum - 1;
    }
    if (findFlag == false) {
        usedCoreNum = 1;
    }

    int64_t blockFactor = totalNum / usedCoreNum;

    // 计算UB上可用的最大内存
    constexpr uint64_t UB_RESERVED_BYTES = 8 * 1024;
    uint64_t usableUb;
    if (ubSize > UB_RESERVED_BYTES) {
        usableUb = ubSize - UB_RESERVED_BYTES;
    } else {
        usableUb = ubSize;
    }

    // 双缓冲，4块缓冲区空间
    int64_t maxUbFactor = static_cast<int64_t>(usableUb / (4ULL * static_cast<uint64_t>(typeSize)));
    maxUbFactor = (maxUbFactor / alignNum) * alignNum;

    if (maxUbFactor < alignNum) {
        maxUbFactor = alignNum;
    }

    int64_t ubFactor;
    if (blockFactor < maxUbFactor) {
        ubFactor = blockFactor;
    } else {
        ubFactor = maxUbFactor;
    }

    ubFactor = (ubFactor / alignNum) * alignNum;
    if (ubFactor <= 0) {
        ubFactor = alignNum;
    }

    // 将分块参数写入tiling结构体
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // 设置并行核数量
    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 根据数据类型选择tilingkey
    uint64_t tilingKey;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);
    }
    context->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForRelu([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct ReluCompileInfo {};

IMPL_OP_OPTILING(Relu)
    .Tiling(ReluTilingFunc)
    .TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
