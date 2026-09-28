/**
 * @file add_custom.cpp
 * @brief add_custom 算子的 Host 侧实现：算子原型注册 + Shape/DataType 推导 + Tiling。
 *
 * 算子语义： z = x + y      （逐元素）
 * 输入输出： x, y, z 均为 2 维 [N2, N1]、float16、ND
 * 目标平台： Ascend 910B / Atlas 800T A2（CANN 8.x）
 */

#include <cstdint>

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "add_custom_tiling.h"

namespace optiling {

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
constexpr uint32_t DTYPE_BYTES  = 2U;    // float16 单元素字节数
constexpr uint32_t ALIGN_BYTES  = 32U;   // UB / GM 搬运的对齐粒度
constexpr uint32_t ALIGN_NUM    = ALIGN_BYTES / DTYPE_BYTES;  // 对齐后的元素数 = 16
constexpr uint32_t BUFFER_NUM   = 2U;    // 双缓冲：VECIN / VECOUT 各 2 份
constexpr uint32_t BUFFER_CNT   = 3U;    // 需要 3 块 UB：x、y、z
constexpr uint64_t UB_RESERVED_BYTES = 8UL * 1024UL;  // UB 预留量，避免把 UB 撑满
constexpr uint32_t EXPECT_DIM_NUM = 2U;  // 需求约定：仅支持 2 维张量 [N2, N1]

// ---------------------------------------------------------------------------
// Tiling 主逻辑
// ---------------------------------------------------------------------------
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    AddCustomTilingData tiling;

    // ---- 1. 取平台信息 ----------------------------------------------------
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // 该算子只用 Vector 单元，按 AIV 核数切分；不要用 GetCoreNum()（那是 AIC 核数）。
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    if (coreNum == 0U) {
        coreNum = 1U;
    }

    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (ubSize == 0UL) {
        OP_LOGE(context->GetNodeName(), "AddCustom: failed to get UB size.");
        return ge::GRAPH_FAILED;
    }

    // ---- 2. 校验输入 ------------------------------------------------------
    gert::TilingData* rawTiling = context->GetRawTilingData();
    if (rawTiling == nullptr) {
        OP_LOGE(context->GetNodeName(), "AddCustom: raw tiling data is null.");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* yShape = context->GetInputShape(1);
    const gert::StorageShape* zShape = context->GetOutputShape(0);
    if (xShape == nullptr || yShape == nullptr || zShape == nullptr) {
        OP_LOGE(context->GetNodeName(), "AddCustom: input or output shape is null.");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& xStorageShape = xShape->GetStorageShape();
    const gert::Shape& yStorageShape = yShape->GetStorageShape();
    const gert::Shape& zStorageShape = zShape->GetStorageShape();
    const size_t dimNum = xStorageShape.GetDimNum();

    // 需求约定：输入为 2 维张量 [N2, N1]
    if (dimNum != EXPECT_DIM_NUM) {
        OP_LOGE(context->GetNodeName(),
                "AddCustom: only 2-D input [N2, N1] is supported, but got %zu dims.", dimNum);
        return ge::GRAPH_FAILED;
    }

    // x、y、z 三者 shape 必须完全一致（本算子不做广播）
    if (yStorageShape.GetDimNum() != dimNum || zStorageShape.GetDimNum() != dimNum) {
        OP_LOGE(context->GetNodeName(), "AddCustom: rank of x, y and z must be identical.");
        return ge::GRAPH_FAILED;
    }
    for (size_t i = 0U; i < dimNum; ++i) {
        if (xStorageShape.GetDim(i) != yStorageShape.GetDim(i) ||
            xStorageShape.GetDim(i) != zStorageShape.GetDim(i)) {
            OP_LOGE(context->GetNodeName(),
                    "AddCustom: shape of x, y and z must be identical (no broadcast), dim %zu differs.", i);
            return ge::GRAPH_FAILED;
        }
    }

    const int64_t rowNum = xStorageShape.GetDim(0);
    const int64_t colNum = xStorageShape.GetDim(1);
    if (rowNum < 0 || colNum < 0 ||
        rowNum > static_cast<int64_t>(UINT32_MAX) || colNum > static_cast<int64_t>(UINT32_MAX)) {
        OP_LOGE(context->GetNodeName(), "AddCustom: invalid dim value [%ld, %ld].",
                static_cast<long>(rowNum), static_cast<long>(colNum));
        return ge::GRAPH_FAILED;
    }
    const uint64_t totalLength64 = static_cast<uint64_t>(rowNum) * static_cast<uint64_t>(colNum);

    // ---- 3. 空张量直接返回 ------------------------------------------------
    // tiling 字段全部置 0，kernel 侧解析到 tileCount == 0 时不会进入任何循环。
    if (totalLength64 == 0U) {
        tiling.set_rows(static_cast<uint32_t>(rowNum));
        tiling.set_cols(static_cast<uint32_t>(colNum));
        tiling.set_totalLength(0U);
        tiling.set_tileNum(0U);
        tiling.set_tileCount(0U);
        tiling.set_perCoreTile(0U);
        context->SetBlockDim(1U);
        context->GetWorkspaceSizes(1)[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
        tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
        rawTiling->SetDataSize(tiling.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    if (totalLength64 > static_cast<uint64_t>(UINT32_MAX)) {
        OP_LOGE(context->GetNodeName(), "AddCustom: element count %lu exceeds the uint32 range.",
                static_cast<unsigned long>(totalLength64));
        return ge::GRAPH_FAILED;
    }
    const uint32_t totalLength = static_cast<uint32_t>(totalLength64);

    // ---- 4. 计算 tile 粒度 ------------------------------------------------
    // UB 中同时驻留 BUFFER_NUM * BUFFER_CNT 块 tile：
    //   VECIN  : x、y 各 BUFFER_NUM 份
    //   VECOUT : z    BUFFER_NUM 份
    // 因此单 tile 的最大字节数为 (ubSize - 预留) / (BUFFER_NUM * BUFFER_CNT)。
    // 预留 8KB：TPipe 内部管理需要占用少量 UB，且把 UB 撑到 100% 会被框架的
    // buffer 校验直接拒掉。
    const uint64_t usableUbSize = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;
    uint32_t tileNumByUb = static_cast<uint32_t>(usableUbSize / (BUFFER_NUM * BUFFER_CNT * DTYPE_BYTES));
    // 向下对齐到 32B，保证每个 tile 的起始地址都落在 32B 边界上
    tileNumByUb = tileNumByUb / ALIGN_NUM * ALIGN_NUM;
    if (tileNumByUb < ALIGN_NUM) {
        tileNumByUb = ALIGN_NUM;  // 兜底，避免 tile 长度为 0
    }

    // 若整张张量还装不满一个 tile，就把 tile 缩到实际长度（同样 32B 向上对齐）
    uint32_t tileNum = tileNumByUb;
    if (totalLength < tileNum) {
        tileNum = (totalLength + ALIGN_NUM - 1U) / ALIGN_NUM * ALIGN_NUM;
    }
    if (tileNum == 0U) {
        tileNum = ALIGN_NUM;
    }

    // ---- 5. 按 tile 把任务切给各 AI Core ----------------------------------
    const uint32_t tileCount = (totalLength + tileNum - 1U) / tileNum;
    uint32_t blockDim = (coreNum < tileCount) ? coreNum : tileCount;
    if (blockDim == 0U) {
        blockDim = 1U;
    }
    const uint32_t perCoreTile = (tileCount + blockDim - 1U) / blockDim;

    // ---- 6. 落盘 tiling ---------------------------------------------------
    tiling.set_rows(static_cast<uint32_t>(rowNum));
    tiling.set_cols(static_cast<uint32_t>(colNum));
    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);
    tiling.set_tileCount(tileCount);
    tiling.set_perCoreTile(perCoreTile);

    context->SetBlockDim(blockDim);

    // 系统预留 workspace（框架 API 内部使用），本算子自身不需要额外 workspace
    context->GetWorkspaceSizes(1)[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    tiling.SaveToBuffer(rawTiling->GetData(), rawTiling->GetCapacity());
    rawTiling->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

#if defined(ENABLE_TILING_PARSE)
// 仅在使用旧版 TilingPrepare 注册方式（IMPL_OP_OPTILING）时需要，默认关闭。
static ge::graphStatus TilingPrepare(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}
#endif

}  // namespace optiling

// ---------------------------------------------------------------------------
// Shape / DataType 推导（GE 图模式编译期使用）
// ---------------------------------------------------------------------------
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* zShape = context->GetOutputShape(0);
    if (xShape == nullptr || zShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // z 与 x 同 shape
    *zShape = *xShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

// ---------------------------------------------------------------------------
// 算子原型注册
// ---------------------------------------------------------------------------
namespace ops {
class AddCustom : public OpDef {
public:
    explicit AddCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        // 目标平台：Ascend 910B 系列
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(AddCustom);
}  // namespace ops
