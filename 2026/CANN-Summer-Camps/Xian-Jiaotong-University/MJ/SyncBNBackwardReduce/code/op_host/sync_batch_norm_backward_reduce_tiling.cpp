/*!
 * \file sync_batch_norm_backward_reduce_tiling.cpp
 * \brief SyncBatchNormBackwardReduce 算子 Tiling 实现
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/sync_batch_norm_backward_reduce_tiling_data.h"
#include "../op_kernel/sync_batch_norm_backward_reduce_tiling_key.h"
#include <graph/utils/type_utils.h>
#include <algorithm>
namespace optiling
{

    using Ops::Base::CeilAlign;
    using Ops::Base::CeilDiv;
    using Ops::Base::FloorAlign;
    using Ops::Base::FloorDiv;
    using Ops::Base::GetUbBlockSize;

    constexpr uint32_t WS_SYS_SIZE = 0U;
    constexpr int64_t MIN_SPLIT_THRESHOLD = 1024;
    constexpr int64_t BLOCK_SIZE = 32;

    static const gert::Shape g_vec_1_shape = {1};

    static inline const gert::Shape EnsureNotScalar(const gert::Shape &in_shape)
    {
        if (in_shape.GetDimNum() == 0)
        {
            return g_vec_1_shape;
        }
        return in_shape;
    }

    static ge::graphStatus GetPlatformInfo(gert::TilingContext *context, uint64_t &ubSize, int64_t &coreNum)
    {
        fe::PlatFormInfos *platformInfoPtr = context->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        OP_CHECK_IF(coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize is 0"), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus GetWorkspaceSize(gert::TilingContext *context)
    {
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
        currentWorkspace[0] = WS_SYS_SIZE;
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus SyncBatchNormBackwardReduceTilingFunc(gert::TilingContext *context)
    {
        // 1. 获取输入信息
        auto &inShape = EnsureNotScalar(context->GetInputShape(0)->GetStorageShape());
        int64_t totalNum = inShape.GetShapeSize();
        ge::DataType dtype = context->GetInputDesc(0)->GetDataType();
        uint32_t typeLen;
        ge::TypeUtils::GetDataTypeLength(dtype, typeLen);

        // 2. 获取硬件信息
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

        // 3. 计算 ubFactor
        int64_t elemPerBlock = BLOCK_SIZE / typeLen;
        int64_t bytesPerElem = (dtype == ge::DT_FLOAT) ? 48 : 72;
        int64_t ubFactor = (static_cast<int64_t>(ubSize) / bytesPerElem / elemPerBlock) * elemPerBlock;
        if (ubFactor <= 0)
            ubFactor = elemPerBlock;

        // 4. 计算核数和 blockFactor
        int64_t totalBlocks = CeilDiv(totalNum, elemPerBlock);
        int64_t useCoreNum = coreNum < totalBlocks ? coreNum : totalBlocks;
        if (totalNum < MIN_SPLIT_THRESHOLD)
            useCoreNum = 1;
        int64_t blockFactor = CeilDiv(totalNum, useCoreNum);
        blockFactor = CeilAlign(std::max(blockFactor, elemPerBlock), elemPerBlock);

        // 5. 设置 Tiling 数据和 Key
        auto *tiling = context->GetTilingData<SyncBatchNormBackwardReduceTilingData>();
        OP_CHECK_NULL_WITH_CONTEXT(context, tiling);

        tiling->totalNum = totalNum;
        tiling->blockFactor = blockFactor;
        tiling->ubFactor = ubFactor;
        context->SetBlockDim(useCoreNum);
        uint64_t tilingKey;
        if (dtype == ge::DT_FLOAT16)
        {
            tilingKey = GET_TPL_TILING_KEY(SYNCBATCHNORMBACKWARDREDUCE_TPL_SCH_MODE_0);
        }
        else if (dtype == ge::DT_FLOAT)
        {
            tilingKey = GET_TPL_TILING_KEY(SYNCBATCHNORMBACKWARDREDUCE_TPL_SCH_MODE_1);
        }
        else if (dtype == ge::DT_BF16)
        {
            tilingKey = GET_TPL_TILING_KEY(SYNCBATCHNORMBACKWARDREDUCE_TPL_SCH_MODE_2);
        }
        context->SetTilingKey(tilingKey);
        return ge::GRAPH_SUCCESS;
    }

    static ge::graphStatus TilingParseForSyncBatchNormBackwardReduce([[maybe_unused]] gert::TilingParseContext *context)
    {
        return ge::GRAPH_SUCCESS;
    }

    struct SyncBatchNormBackwardReduceCompileInfo
    {
    };

    IMPL_OP_OPTILING(SyncBatchNormBackwardReduce).Tiling(SyncBatchNormBackwardReduceTilingFunc).TilingParse<SyncBatchNormBackwardReduceCompileInfo>(TilingParseForSyncBatchNormBackwardReduce);

} // namespace optiling