/**
 * @file atanh_tiling.cpp
 * @brief Atanh 算子 Tiling 逻辑（host 侧）
 *
 * 职责：
 *   1. 将 4 维 shape [N4,N3,N2,N1] 展平，得到总元素个数 totalLength；
 *   2. 依据硬件可用核数与数据量确定实际启动核数 blockDim；
 *   3. 将 tiling 数据写入 buffer 供 kernel 读取。
 */
#include "atanh_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

// 每个 tile 处理的元素个数（half 类型，2B/元素）
// 2048 * 2B = 4KB，为 32B 的整数倍，满足 DataCopy 对齐要求
constexpr uint32_t TILE_LENGTH = 2048;

static ge::graphStatus AtanhTilingFunc(gert::TilingContext* context) {
    // ---------- 1. 计算总元素个数（任意维 shape 展平） ----------
    const gert::Shape* inputShape = context->GetInputShape(0);
    uint64_t totalLength64 = 1;
    for (size_t i = 0; i < inputShape->GetDimNum(); i++) {
        totalLength64 *= static_cast<uint64_t>(inputShape->GetDim(i));
    }
    if (totalLength64 == 0) {
        // 空 tensor：设置 blockDim=1，kernel 不执行有效计算
        context->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }
    uint32_t totalLength = static_cast<uint32_t>(totalLength64);

    // ---------- 2. 获取硬件可用核数 ----------
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t coreNum = platform.GetCoreNum();

    // ---------- 3. 确定实际核数：每个核至少分配一个 tile 的数据 ----------
    uint32_t needCore = static_cast<uint32_t>(
        (totalLength64 + TILE_LENGTH - 1) / TILE_LENGTH);
    uint32_t blockDim = (coreNum < needCore) ? coreNum : needCore;
    if (blockDim == 0) {
        blockDim = 1;
    }

    // ---------- 4. 保存 tiling 数据并设置 blockDim ----------
    AtanhTilingData tilingData;
    tilingData.set_totalLength(totalLength);
    tilingData.set_blockDim(blockDim);

    context->SetBlockDim(blockDim);
    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

REGISTER_TILING(atanh, optiling::AtanhTilingFunc);
