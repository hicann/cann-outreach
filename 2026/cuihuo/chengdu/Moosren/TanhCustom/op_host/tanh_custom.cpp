
#include "../op_kernel/tanh_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {
const uint32_t BLOCK_DIM = 8;
const uint32_t TILE_NUM  = 8;

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    auto x_shape = context->GetInputShape(0);
    if (x_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    uint32_t totalLength = static_cast<uint32_t>(x_shape->GetShape().GetShapeSize());
    if (totalLength == 0) {
        return ge::GRAPH_FAILED;
    }

    // ---- 获取数据类型大小（FP16 = 2 字节） ----
    auto dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dtypeSize = (dtype == ge::DT_FLOAT) ? 4u : 2u;  // FP16=2, FP32=4

    // ---- 获取硬件参数（动态查询，不硬编码） ----
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // 实际可用核数
    uint32_t coreNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    if (coreNum == 0) {
        coreNum = BLOCK_DIM;  // 回退值
    }

    // UB 可用大小（字节）
    uint64_t u_bSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, u_bSize);
    if (u_bSize == 0) {
        u_bSize = 256 * 1024;  // 回退值 256KB
    }

    // ---- 核间切分（Block Tiling） ----
    // 每核负责的元素数，向上取整后按 32B（FP16: 16 元素）对齐
    uint32_t alignNum     = 32u / dtypeSize;          // FP16: 16, FP32: 8
    uint32_t blockLength  = (totalLength + coreNum - 1) / coreNum;
    blockLength           = ((blockLength + alignNum - 1) / alignNum) * alignNum;

    // 实际使用的核数（最后一核可能无数据则不启动）
    uint32_t usedCoreNum  = (totalLength + blockLength - 1) / blockLength;
    uint32_t blockDim     = usedCoreNum;

    // 合计：bufferCoefficient = 2*2*dtypeSize + 3*4
    uint32_t bufferCoefficient = 2u * 2u * dtypeSize + 3u * 4u;
    uint32_t maxTileElements   = static_cast<uint32_t>(u_bSize) / bufferCoefficient;
    // 对齐到 32B 边界
    uint32_t tileLength = (maxTileElements / alignNum) * alignNum;
    if (tileLength == 0) {
        tileLength = alignNum;  // 至少一个对齐块
    }

    // 每核内 tile 分块数量（向上取整）
    uint32_t tileNum = (blockLength + tileLength - 1) / tileLength;
    if (tileNum == 0) {
        tileNum = TILE_NUM;
    }

    // ---- 写入 TilingData ----
    TanhCustomTilingData tilingData;
    tilingData.totalLength  = totalLength;
    tilingData.blockLength  = blockLength;
    tilingData.tileLength   = tileLength;
    tilingData.tileNum      = tileNum;

    context->SetBlockDim(blockDim);

    auto tilingDataPtr = context->GetRawTilingData();
    tilingDataPtr->Append(tilingData);

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class TanhCustom : public OpDef {
public:
    explicit TanhCustom(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

 OP_ADD(TanhCustom);
}

