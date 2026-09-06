// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // 获取平台信息
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t numCoresAiv = platform.GetCoreNumAiv();

    uint64_t ubSize = 0;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    // 获取输入Tensor信息
    const gert::Tensor *tensorX =
        context->GetRequiredInputTensor(0);

    ge::DataType dtypeX =
        tensorX->GetDataType();

    uint32_t dtypeSize =
        static_cast<uint32_t>(
            ge::GetSizeByDataType(dtypeX));

    uint32_t totalLength =
        static_cast<uint32_t>(
            tensorX->GetShapeSize());

    // 配置TilingKey
    uint32_t DT_X =
        static_cast<uint32_t>(dtypeX);

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // =========================================================
    // 1. 多核切分
    // =========================================================

    uint32_t coreNum =
        (numCoresAiv > 0)
            ? static_cast<uint32_t>(numCoresAiv)
            : 1U;

    // 避免核数超过元素数量
    if (totalLength > 0 && coreNum > totalLength) {
        coreNum = totalLength;
    }

    if (totalLength == 0) {
        coreNum = 1;
    }

    uint32_t blockLength = 0;

    if (totalLength > 0) {
        // 每个核最多负责的元素数
        blockLength =
            (totalLength + coreNum - 1U) / coreNum;

        // 重新计算真正需要的核数
        coreNum =
            (totalLength + blockLength - 1U)
            / blockLength;
    }

    // =========================================================
    // 2. 核内Tile切分
    //
    // UB中需要同时存放：
    // xLocal
    // yLocal
    // zLocal
    //
    // 因此单个Tensor最多使用约1/4 UB，
    // 给系统和队列预留一部分空间。
    // =========================================================

    uint64_t maxTileBytes = ubSize / 4U;

    uint32_t maxTileLength = 1;

    if (dtypeSize > 0) {
        maxTileLength =
            static_cast<uint32_t>(
                maxTileBytes / dtypeSize);

        if (maxTileLength == 0) {
            maxTileLength = 1;
        }
    }

    uint32_t tileLength = 1;

    if (blockLength > 0) {
        tileLength =
            (blockLength < maxTileLength)
                ? blockLength
                : maxTileLength;
    }

    // =========================================================
    // 3. 写入TilingData
    // =========================================================

    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->totalLength = totalLength;
    tiling->blockLength = blockLength;
    tiling->tileLength = tileLength;

    // 设置启动核数
    context->SetBlockDim(coreNum);

    // 本算子不需要额外Workspace
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const gert::Shape *xShape =
        context->GetInputShape(0);

    gert::Shape *zShape =
        context->GetOutputShape(0);

    // z的shape与x完全一致
    *zShape = *xShape;

    return GRAPH_SUCCESS;
}


static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    ge::DataType inputDataType =
        context->GetInputDataType(0);

    // z的数据类型与x一致
    context->SetOutputDataType(
        0,
        inputDataType);

    return GRAPH_SUCCESS;
}

}  // namespace ge


namespace ops {

class Mul : public OpDef {
public:
    explicit Mul(const char *name)
        : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops