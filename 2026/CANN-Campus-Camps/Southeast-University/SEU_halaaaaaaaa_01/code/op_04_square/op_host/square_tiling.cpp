/*!
 * \file square_tiling.cpp
 * \brief Square 算子 tiling 实现
 */

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/square_tiling_data.h"
#include "../op_kernel/square_tiling_key.h"

namespace optiling {

static ge::graphStatus SquareTilingFunc(gert::TilingContext *context)
{
    // =========================================================
    // 1. 获取输入总元素数量
    // 支持任意 (..., N) shape
    // =========================================================
    int64_t totalNum =
        context->GetInputShape(0)
            ->GetOriginShape()
            .GetShapeSize();

    if (totalNum <= 0) {
        return ge::GRAPH_FAILED;
    }

    // =========================================================
    // 2. 获取输入数据类型
    // =========================================================
    ge::DataType dtype =
        context->GetInputDesc(0)->GetDataType();

    // =========================================================
    // 3. 获取平台 Vector Core 数量
    // =========================================================
    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    uint32_t coreNum =
        platform.GetCoreNumAiv();

    if (coreNum == 0) {
        coreNum = 1;
    }

    // 数据很少时，不启动超过元素数的 Core
    if (static_cast<int64_t>(coreNum) > totalNum) {
        coreNum =
            static_cast<uint32_t>(totalNum);
    }

    // =========================================================
    // 4. 计算每核处理元素数量
    //
    // 采用 ceilDiv，最后一个 Core 自动处理尾块
    // =========================================================
    int64_t blockFactor =
        (totalNum +
         static_cast<int64_t>(coreNum) - 1) /
        static_cast<int64_t>(coreNum);

    /*
     * 重新计算实际 blockDim。
     *
     * 例如：
     * totalNum = 10
     * coreNum = 8
     *
     * blockFactor = 2
     *
     * 实际只需要：
     * ceil(10 / 2) = 5 个 Core
     *
     * 避免启动没有数据的 Core。
     */
    uint32_t blockDim =
        static_cast<uint32_t>(
            (totalNum + blockFactor - 1) /
            blockFactor);

    context->SetBlockDim(blockDim);

    // =========================================================
    // 5. 计算每次 UB 处理的数据量
    //
    // 使用最多 4096 个元素一个 Tile。
    // float:
    //   4096 * 4 = 16 KB
    //
    // half:
    //   4096 * 2 = 8 KB
    //
    // 再使用 DoubleBuffer，UB 空间仍然很安全。
    // =========================================================
    constexpr int64_t MAX_UB_FACTOR = 4096;

    int64_t ubFactor = blockFactor;

    if (ubFactor > MAX_UB_FACTOR) {
        ubFactor = MAX_UB_FACTOR;
    }

    /*
     * UB Tensor 的起始地址要求 32B 对齐。
     *
     * float32:
     * 32 / 4 = 8 elements
     *
     * float16:
     * 32 / 2 = 16 elements
     *
     * 因此将 UB Buffer 容量向上对齐。
     *
     * 注意：
     * 实际搬运数量仍然由 currentNum 控制，
     * 所以即使 ubFactor 比实际尾块大，也不会越界。
     */
    int64_t alignNum = 8;

    if (dtype == ge::DT_FLOAT16) {
        alignNum = 16;
    }

    ubFactor =
        ((ubFactor + alignNum - 1) /
         alignNum) *
        alignNum;

    // =========================================================
    // 6. 写入 TilingData
    // =========================================================
    SquareTilingData *tiling =
        context->GetTilingData<SquareTilingData>();

    tiling->totalNum = totalNum;
    tiling->blockFactor = blockFactor;
    tiling->ubFactor = ubFactor;

    // =========================================================
    // 7. 设置 TilingKey
    //
    // MODE_0 -> half
    // MODE_1 -> float
    // =========================================================
    uint32_t schMode =
        SQUARE_TPL_SCH_MODE_0;

    if (dtype == ge::DT_FLOAT16) {
        schMode =
            SQUARE_TPL_SCH_MODE_0;
    } else {
        schMode =
            SQUARE_TPL_SCH_MODE_1;
    }

    ASCENDC_TPL_SEL_PARAM(
        context,
        schMode);

    // =========================================================
    // 8. Square 不需要额外 Workspace
    // =========================================================
    size_t *workspaceSize =
        context->GetWorkspaceSizes(1);

    workspaceSize[0] = 0;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Square)
    .Tiling(SquareTilingFunc);

} // namespace optiling