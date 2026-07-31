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
// 在文件开头添加辅助函数
static uint32_t GetDataTypeSize(ge::DataType dtype) {
    switch(dtype) {
        case ge::DT_FLOAT16: return 2;
        case ge::DT_FLOAT: return 4;
        case ge::DT_INT8: return 1;
        case ge::DT_INT16: return 2;
        case ge::DT_INT32: return 4;
        case ge::DT_INT64: return 8;
        case ge::DT_UINT8: return 1;
        case ge::DT_UINT16: return 2;
        case ge::DT_UINT32: return 4;
        case ge::DT_UINT64: return 8;
        case ge::DT_BOOL: return 1;
        case ge::DT_DOUBLE: return 8;
        default: return 4;
    }
}


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
  
    //d TODO: 实现 Tiling 逻辑
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
     //获取输入信息
    auto inputDesc = context->GetInputDesc(0);
    auto inputShape = context->GetInputShape(0);
    auto originShape = inputShape->GetOriginShape();
    int64_t totalNum = originShape.GetShapeSize();
    
    //printf("ubSize=%lu, coreNum=%ld\n", ubSize, coreNum);
   // 获取输入shape信息
     if (totalNum == 0) {
        tiling->totalNum = 0;
        tiling->blockFactor = 1;
        tiling->ubFactor = 0;
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t blockDim, blockFactor;
    constexpr int64_t SINGLE_CORE_THRESHOLD = 128;
    
    if (totalNum <= SINGLE_CORE_THRESHOLD) {
        // 小数据量单核处理
        blockDim = 1;
        blockFactor = totalNum;
    } else {
        // 计算每个核处理的数据量，考虑核心数
        blockFactor = (totalNum + coreNum - 1) / coreNum;
        // 对齐到64的倍数（假设需要64字节对齐）
        constexpr int64_t ALIGN_SIZE = 64;
        blockFactor = ((blockFactor + ALIGN_SIZE - 1) / ALIGN_SIZE) * ALIGN_SIZE;
        // 实际使用的核数
        blockDim = (totalNum + blockFactor - 1) / blockFactor;
        // 限制核数不超过可用核心数
        blockDim = std::min(blockDim, coreNum);
        // 重新计算blockFactor
        blockFactor = (totalNum + blockDim - 1) / blockDim;
    }
    
    
    ge::DataType dtype = inputDesc->GetDataType();
    //uint32_t typeByte =  gert::GetSizeOfDataType(dtype); // float16=2, float=4
    uint32_t typeByte = GetDataTypeSize(dtype);
    if (typeByte == 0) {
        OP_LOGE(context, "Invalid data type");
        return ge::GRAPH_FAILED;
    }
    // 统计所有UB张量单元素总字节（根据你Compute里开辟的LocalTensor统计）
    uint32_t elemByte = typeByte * 2 + 4 * 4 + sizeof(int32_t);
    elemByte = ((elemByte + 63) / 64) * 64;
    // 单次UB最大可处理元素个数 
    int64_t ubFactor = ubSize / elemByte;
    // 保证至少处理一条数据
    ubFactor = std::max(ubFactor, static_cast<int64_t>(1));
    
    // 对齐单位：32 / typeByte 个元素
    constexpr int64_t ALIGN_32B = 32;
    int64_t alignCount = ALIGN_32B / typeByte;  // half:16, float:8, int32:8
    ubFactor = ((ubFactor + alignCount - 1) / alignCount) * alignCount;
    ubFactor = std::max(ubFactor, alignCount);
    
    if (ubFactor > blockFactor) {
        // 如果blockFactor小于alignCount，直接使用blockFactor
        if (blockFactor < alignCount) {
            ubFactor = blockFactor;
        } else {
            // 否则对齐到alignCount的倍数，但不超过blockFactor
            ubFactor = ((blockFactor / alignCount) * alignCount);
            if (ubFactor == 0) {
                ubFactor = alignCount;
            }
        }
    }
    
    //d TODO: 设置 tiling 数据
    
    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;


    context->SetBlockDim(blockDim);

    // 根据输入 dtype 选择 tilingKey
    uint64_t tilingKey;
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
