/**
 * @file inplace_rsqrt.cpp
 * @brief InplaceRsqrt 算子 host 侧 tiling（自定义算子包 / 框架调用方式）
 */

#include "inplace_rsqrt_tiling.h"

#include <cstdint>
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// 单次搬运的元素个数：8192 * 2Byte = 16KB
// UB 占用 = BUFFER_NUM(2) * 16KB = 32KB（打开牛顿迭代时再 + 2 * 16KB）
constexpr uint32_t TILE_LENGTH = 8192;

static inline uint32_t CeilDivU32Host(uint32_t a, uint32_t b)
{
    return (b == 0) ? a : (a + b - 1) / b;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *inputShape = context->GetInputShape(0);
    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 4 维 [N4, N3, N2, N1] 展平成一维：元素级算子只关心总元素个数
    uint64_t shapeSize = static_cast<uint64_t>(inputShape->GetOriginShape().GetShapeSize());
    if (shapeSize > static_cast<uint64_t>(UINT32_MAX)) {
        return ge::GRAPH_FAILED;  // 单次下发按 uint32 寻址，超出范围直接拒绝
    }
    uint32_t totalLength = static_cast<uint32_t>(shapeSize);

    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    uint32_t aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (aivNum == 0) {
        aivNum = 1;
    }

    // 数据量小时不占满所有核：每核约一个 tile；空张量下发 1 个核，kernel 侧直接返回
    uint32_t blockDim = CeilDivU32Host(totalLength, TILE_LENGTH);
    if (blockDim == 0) {
        blockDim = 1;
    }
    if (blockDim > aivNum) {
        blockDim = aivNum;
    }
    context->SetBlockDim(blockDim);

    // 无论数据量多少都要把 tiling 写入，避免 kernel 侧读到未初始化数据
    InplaceRsqrtTilingData tiling;
    tiling.set_totalLength(totalLength);
    tiling.set_tileLength(TILE_LENGTH);
    if (context->GetRawTilingData() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 原地算子：不需要 workspace，也不额外申请输出显存
    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces != nullptr) {
        workspaces[0] = 0;
    }
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

IMPL_OP_OPTILING(InplaceRsqrt).Tiling(optiling::TilingFunc);