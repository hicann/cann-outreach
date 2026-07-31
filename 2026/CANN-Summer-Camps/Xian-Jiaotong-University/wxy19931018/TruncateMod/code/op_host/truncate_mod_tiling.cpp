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

    // 1. 获取输出描述
    auto outputDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputDesc);

    // 2. 获取 Shape 并提取 StorageShape 传入 EnsureNotScalar
    const gert::StorageShape* outputShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    gert::Shape ensuredShape = EnsureNotScalar(outputShape->GetStorageShape());
    int64_t totalNum = ensuredShape.GetShapeSize();
    OP_CHECK_IF(totalNum <= 0, OP_LOGE(context, "totalNum is invalid"), return ge::GRAPH_FAILED);

    // 3. 从 outputDesc 获取数据类型
    ge::DataType dtype = outputDesc->GetDataType();

    int64_t typeSize = TYPE_SIZE;
    if (dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16) {
        typeSize = 2;
    } else if (dtype == ge::DT_FLOAT) {
        typeSize = 4;
    } else if (dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
        typeSize = 1;
    } else if (dtype == ge::DT_INT32) {
        typeSize = 4;
    } else if (dtype == ge::DT_INT64) {
        typeSize = 8;
    }

    // 计算 UB 可以容纳的元素数量（考虑需要 x1, x2, y 三个缓冲区）
    int64_t ubElementNum = static_cast<int64_t>(ubSize) / (typeSize * 3);
    int64_t ubFactor = FloorAlign<int64_t>(ubElementNum, 32);

    if (ubFactor <= 0) {
        ubFactor = 32;
    }

    // 计算每个核处理的数据量
    int64_t blockFactor = CeilDiv(totalNum, coreNum);
    if (blockFactor < MIN_SPLIT_THRESHOLD) {
        blockFactor = totalNum;
        coreNum = 1;
    }

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    context->SetBlockDim(static_cast<uint32_t>(coreNum));

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc != nullptr && (inputDesc->GetDataType() == ge::DT_FLOAT16 || inputDesc->GetDataType() == ge::DT_BF16)) {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_0);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TRUNCATEMOD_TPL_SCH_MODE_1);
    }
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
