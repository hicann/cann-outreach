#include "../op_kernel/add_custom_template_tiling.h"
#include "register/op_def_registry.h"

#include <algorithm>
#include <cstdint>

namespace optiling {

namespace {

// Ascend 910B 上本算子最多使用的核数。
constexpr uint32_t MAX_BLOCK_DIM = 40;

// 每个 tile 的目标字节数。
// Add 属于访存密集型算子，先以 16 KB 作为初始调优值。
constexpr uint32_t TARGET_TILE_BYTES = 16 * 1024;

// DataCopy 的基础块大小为 32 Byte。
constexpr uint32_t DATA_COPY_ALIGN_BYTES = 32;

/**
 * 找出不超过 maxBlockDim 的最大核数，使：
 *
 * 1. totalLength 可以被 blockDim 整除；
 * 2. 每核元素数满足 32 Byte 对齐；
 *
 * 这样 Kernel 不需要额外处理核间尾块。
 */
static uint32_t SelectBlockDim(
    uint32_t totalLength,
    uint32_t maxBlockDim,
    uint32_t alignElements)
{
    uint32_t upper = std::min(totalLength, maxBlockDim);

    for (uint32_t blockDim = upper; blockDim > 0; --blockDim) {
        if (totalLength % blockDim != 0) {
            continue;
        }

        const uint32_t blockLength = totalLength / blockDim;

        if (blockLength % alignElements == 0) {
            return blockDim;
        }
    }

    return 1;
}

/**
 * 根据目标 tile 大小选择 tileNum。
 *
 * 要求：
 *
 * 1. tileNum 可以整除 blockLength；
 * 2. tileLength 满足 32 Byte 对齐；
 * 3. tileLength 尽量接近 targetTileLength。
 */
static uint32_t SelectTileNum(
    uint32_t blockLength,
    uint32_t targetTileLength,
    uint32_t alignElements)
{
    if (blockLength <= targetTileLength) {
        return 1;
    }

    uint32_t estimatedTileNum =
        (blockLength + targetTileLength - 1) / targetTileLength;

    if (estimatedTileNum == 0) {
        estimatedTileNum = 1;
    }

    /*
     * 从估算值开始向上寻找。
     *
     * tileNum 越大，tileLength 越小。
     * 找到的第一个合法值通常最接近目标 tile 大小。
     */
    for (uint32_t tileNum = estimatedTileNum;
         tileNum <= blockLength;
         ++tileNum) {
        if (blockLength % tileNum != 0) {
            continue;
        }

        const uint32_t tileLength = blockLength / tileNum;

        if (tileLength % alignElements == 0) {
            return tileNum;
        }
    }

    /*
     * 正常对齐输入不会进入这里。
     * 退化为单 tile，保持所有数据由当前核处理。
     */
    return 1;
}

}  // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    AddCustomTemplateTilingData* tiling =
        context->GetTilingData<AddCustomTemplateTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape* inputShape = context->GetInputShape(0);

    if (inputShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const int64_t shapeSize =
        inputShape->GetOriginShape().GetShapeSize();

    if (shapeSize < 0 ||
        static_cast<uint64_t>(shapeSize) > UINT32_MAX) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t totalLength =
        static_cast<uint32_t>(shapeSize);

    /*
     * 空 Tensor 不启动大量核。
     */
    if (totalLength == 0) {
        context->SetBlockDim(1);
        tiling->totalLength = 0;
        tiling->tileNum = 1;
        return ge::GRAPH_SUCCESS;
    }

    const ge::DataType dataType =
        context->GetInputDesc(0)->GetDataType();

    uint32_t elementSize = 0;

    if (dataType == ge::DT_FLOAT16) {
        elementSize = sizeof(uint16_t);
    } else if (dataType == ge::DT_FLOAT) {
        elementSize = sizeof(float);
    } else {
        return ge::GRAPH_FAILED;
    }

    const uint32_t alignElements =
        DATA_COPY_ALIGN_BYTES / elementSize;

    /*
     * 当前 Kernel 使用普通 DataCopy。
     * 输入总字节数必须满足 32 Byte 对齐。
     *
     * 原实现同样依赖这一条件。
     */
    if (totalLength % alignElements != 0) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t targetTileLength =
        std::max(
            alignElements,
            TARGET_TILE_BYTES / elementSize);

    const uint32_t blockDim =
        SelectBlockDim(
            totalLength,
            MAX_BLOCK_DIM,
            alignElements);

    const uint32_t blockLength =
        totalLength / blockDim;

    const uint32_t tileNum =
        SelectTileNum(
            blockLength,
            targetTileLength,
            alignElements);

    context->SetBlockDim(blockDim);

    tiling->totalLength = totalLength;
    tiling->tileNum = tileNum;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(
    gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const gert::Shape* xShape =
        context->GetInputShape(0);

    const gert::Shape* yShape =
        context->GetInputShape(1);

    gert::Shape* zShape =
        context->GetOutputShape(0);

    if (xShape == nullptr ||
        yShape == nullptr ||
        zShape == nullptr) {
        return GRAPH_FAILED;
    }

    /*
     * Add 要求两个输入 Shape 相同。
     */
    if (*xShape != *yShape) {
        return GRAPH_FAILED;
    }

    *zShape = *xShape;

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(
    gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }

    const ge::DataType xDataType =
        context->GetInputDataType(0);

    const ge::DataType yDataType =
        context->GetInputDataType(1);

    /*
     * 当前 Kernel 不执行类型转换，因此要求输入类型一致。
     */
    if (xDataType != yDataType) {
        return GRAPH_FAILED;
    }

    if (xDataType != ge::DT_FLOAT16 &&
        xDataType != ge::DT_FLOAT) {
        return GRAPH_FAILED;
    }

    context->SetOutputDataType(0, xDataType);

    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class AddCustomTemplate : public OpDef {
public:
    explicit AddCustomTemplate(const char* name)
        : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            })
            .UnknownShapeFormat({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);

        this->AICore()
            .AddConfig("ascend910b");
    }
};

OP_ADD(AddCustomTemplate);

}  // namespace ops