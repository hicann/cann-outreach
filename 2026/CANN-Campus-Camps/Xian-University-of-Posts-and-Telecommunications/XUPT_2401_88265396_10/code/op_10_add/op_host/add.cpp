// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/add_tiling.h"
#include "../op_kernel/tiling_key_add.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    /*
     * ============================================================
     * 1. 获取平台信息
     * ============================================================
     */
    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    int32_t numCoresAiv =
        platform.GetCoreNumAiv();

    uint64_t ubSize = 0;

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize);

    /*
     * ============================================================
     * 2. 获取输入Tensor信息
     * ============================================================
     */
    const gert::Tensor *tensorX =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensorY =
        context->GetRequiredInputTensor(1);

    if (tensorX == nullptr ||
        tensorY == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtypeX =
        tensorX->GetDataType();

    ge::DataType dtypeY =
        tensorY->GetDataType();

    /*
     * Add要求两个输入的数据类型一致。
     */
    if (dtypeX != dtypeY) {
        return ge::GRAPH_FAILED;
    }

    int32_t dtypeSize =
        ge::GetSizeByDataType(dtypeX);

    if (dtypeSize <= 0) {
        return ge::GRAPH_FAILED;
    }

    /*
     * 输入元素总数。
     *
     * 题面给出的shape为：
     *
     * (8, 2048)
     *
     * 即：
     *
     * 8 * 2048 = 16384
     *
     * 这里不写死16384，而是从实际Tensor中读取。
     */
    uint32_t totalLength =
        static_cast<uint32_t>(
            tensorX->GetShapeSize());

    if (totalLength == 0) {
        return ge::GRAPH_FAILED;
    }

    /*
     * ============================================================
     * 3. 配置TilingKey
     * ============================================================
     *
     * 保持原始模板自己的TilingKey机制：
     *
     * float32 -> float模板
     * float16 -> half模板
     */
    uint32_t DT_X =
        static_cast<uint32_t>(dtypeX);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_X);

    /*
     * ============================================================
     * 4. 计算32 Byte对齐元素数
     * ============================================================
     *
     * DataCopy普通搬运要求搬运数据量是32 Byte整数倍。
     *
     * float32:
     *     32 / 4 = 8 elements
     *
     * float16:
     *     32 / 2 = 16 elements
     */
    constexpr uint32_t DATA_BLOCK_SIZE = 32;

    uint32_t alignNum =
        DATA_BLOCK_SIZE /
        static_cast<uint32_t>(dtypeSize);

    /*
     * ============================================================
     * 5. 计算实际使用的AIV Core数量
     * ============================================================
     *
     * 不能简单：
     *
     * SetBlockDim(numCoresAiv)
     *
     * 否则如果Kernel侧没有正确切分，
     * 所有Core可能重复处理同一段数据。
     *
     * 这里寻找满足：
     *
     * 1. totalLength能够被coreNum整除
     * 2. 每个Core负责的数据满足32B对齐
     *
     * 的最大Core数量。
     */
    uint32_t usedCoreNum =
        static_cast<uint32_t>(
            numCoresAiv);

    /*
     * Core数量不能超过实际可切分的数据块数量。
     */
    uint32_t maxCoreByLength =
        totalLength / alignNum;

    if (maxCoreByLength == 0) {
        maxCoreByLength = 1;
    }

    if (usedCoreNum > maxCoreByLength) {
        usedCoreNum = maxCoreByLength;
    }

    if (usedCoreNum == 0) {
        usedCoreNum = 1;
    }

    while (usedCoreNum > 1) {
        if (totalLength % usedCoreNum == 0) {
            uint32_t blockLength =
                totalLength / usedCoreNum;

            if (blockLength % alignNum == 0) {
                break;
            }
        }

        --usedCoreNum;
    }

    uint32_t blockLength =
        totalLength / usedCoreNum;

    /*
     * ============================================================
     * 6. 根据UB大小计算Tile大小
     * ============================================================
     *
     * Kernel中有：
     *
     * input X Queue × 2
     * input Y Queue × 2
     * output Z Queue × 2
     *
     * 总共6份Tile空间。
     *
     * 额外保留8KB UB空间。
     */
    constexpr uint64_t UB_RESERVED_SIZE =
        8 * 1024;

    uint64_t usableUbSize =
        ubSize > UB_RESERVED_SIZE
            ? ubSize - UB_RESERVED_SIZE
            : ubSize;

    uint32_t maxTileLength =
        static_cast<uint32_t>(
            usableUbSize /
            (6ULL *
             static_cast<uint64_t>(
                 dtypeSize)));

    /*
     * Tile同样保证32B对齐。
     */
    maxTileLength =
        (maxTileLength / alignNum) *
        alignNum;

    if (maxTileLength < alignNum) {
        maxTileLength = alignNum;
    }

    uint32_t tileLength =
        blockLength < maxTileLength
            ? blockLength
            : maxTileLength;

    tileLength =
        (tileLength / alignNum) *
        alignNum;

    if (tileLength == 0) {
        tileLength = alignNum;
    }

    /*
     * 防止tileLength超过当前Core负责的数据。
     */
    if (tileLength > blockLength) {
        tileLength = blockLength;
    }

    /*
     * ============================================================
     * 7. 填充TilingData
     * ============================================================
     */
    AddTilingData *tiling =
        context->GetTilingData<AddTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->totalLength =
        totalLength;

    tiling->blockLength =
        blockLength;

    tiling->tileLength =
        tileLength;

    /*
     * ============================================================
     * 8. 设置启动Core数量
     * ============================================================
     */
    context->SetBlockDim(
        usedCoreNum);

    /*
     * ============================================================
     * 9. Workspace
     * ============================================================
     *
     * 当前Add不需要额外Workspace。
     */
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    if (currentWorkspace == nullptr) {
        return ge::GRAPH_FAILED;
    }

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


namespace ge {

/*
 * ================================================================
 * InferShape
 * ================================================================
 *
 * z = x + y
 *
 * 输出Tensor的shape与输入x保持一致。
 */
static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const gert::Shape *xShape =
        context->GetInputShape(0);

    gert::Shape *zShape =
        context->GetOutputShape(0);

    if (xShape == nullptr ||
        zShape == nullptr) {
        return GRAPH_FAILED;
    }

    *zShape = *xShape;

    return GRAPH_SUCCESS;
}


/*
 * ================================================================
 * InferDataType
 * ================================================================
 *
 * 输出z的数据类型与输入x一致。
 */
static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    const auto inputDataType =
        context->GetInputDataType(0);

    return context->SetOutputDataType(
        0,
        inputDataType);
}

}  // namespace ge


namespace ops {

class Add : public OpDef {
public:
    explicit Add(const char *name)
        : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND});

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND});

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT,
                ge::DT_FLOAT16})
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND});

        this->SetInferShape(
                ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);

        this->AICore()
            .SetTiling(
                optiling::TilingFunc)
            .AddConfig(
                "ascend910b");
    }
};

OP_ADD(Add);

}  // namespace ops