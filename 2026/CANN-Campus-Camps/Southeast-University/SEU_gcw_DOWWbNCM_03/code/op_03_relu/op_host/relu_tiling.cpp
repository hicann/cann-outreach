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

const uint32_t BUFFER_NUM = 1;
const uint32_t UB_SPLIT = 1;
const uint32_t UB_RESERVE = 8 * 1024;
const int64_t BLOCK_BYTE_SIZE = 32;
const int64_t BLOCK_DIM_CAP =8;    

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
    // TODO: 实现 Tiling 逻辑
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

    const gert::StorageShape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t totalNum = 1;
    for (size_t i = 0; i < xShape->GetStorageShape().GetDimNum(); i++) {
        totalNum *= xShape->GetStorageShape().GetDim(i);
    }
    if (totalNum < 0) {
        totalNum = 0;
    }

    int64_t typeSize = TYPE_SIZE;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr) {
        int64_t dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(inputDesc->GetDataType()));
        if (dtypeSize > 0) {
            typeSize = dtypeSize;
        }
    }

    int64_t elemPerBlock = (typeSize > 0) ? (BLOCK_BYTE_SIZE / typeSize) : 1;
    if (elemPerBlock < 1) {
        elemPerBlock = 1;
    }

    int64_t blockDim = (coreNum > 0) ? coreNum : 1;
    if (blockDim > totalNum && totalNum > 0) {
        blockDim = totalNum;
    }
    if (blockDim > BLOCK_DIM_CAP) {
        blockDim = BLOCK_DIM_CAP;
    }

    int64_t blockFactor = (totalNum + blockDim - 1) / blockDim;
    blockFactor = (blockFactor + elemPerBlock - 1) / elemPerBlock * elemPerBlock;
    if (blockFactor < elemPerBlock) {
        blockFactor = elemPerBlock;
    }

    uint64_t ubAvailable = (ubSize > UB_RESERVE) ? (ubSize - UB_RESERVE) : (ubSize / 2);
    uint64_t ubMaxElemU =
        ubAvailable / (static_cast<uint64_t>(typeSize) * 2U * static_cast<uint64_t>(BUFFER_NUM));
    int64_t ubMaxElem = static_cast<int64_t>(ubMaxElemU);
    ubMaxElem = ubMaxElem / elemPerBlock * elemPerBlock; // 按 32B 向下对齐
    int64_t ubFactor = blockFactor;
    if (ubFactor > ubMaxElem && ubMaxElem >= elemPerBlock) {
        ubFactor = ubMaxElem;
    }

    ubFactor = (ubFactor / UB_SPLIT) / elemPerBlock * elemPerBlock;
    if (ubFactor < elemPerBlock) {
        ubFactor = elemPerBlock;
    }   

    // TODO: 设置 tiling 数据
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(blockDim));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
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

IMPL_OP_OPTILING(Relu).Tiling(ReluTilingFunc).TilingParse<ReluCompileInfo>(TilingParseForRelu);

} // namespace optiling
