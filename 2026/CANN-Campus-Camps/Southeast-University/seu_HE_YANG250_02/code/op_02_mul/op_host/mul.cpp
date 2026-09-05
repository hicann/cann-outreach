// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t DEFAULT_TILE_NUM = 8;
constexpr uint32_t MAX_USED_CORE_NUM = 8;

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // ================================================================
    // 1. 获取平台信息
    // ================================================================
    auto platform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    int32_t numCoresAiv = platform.GetCoreNumAiv();

    if (numCoresAiv <= 0) {
        return ge::GRAPH_FAILED;
    }

    // ================================================================
    // 2. 获取输入Tensor信息
    // ================================================================
    const gert::Tensor *tensorX =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensorY =
        context->GetRequiredInputTensor(1);

    if (tensorX == nullptr || tensorY == nullptr) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType dtypeX = tensorX->GetDataType();
    ge::DataType dtypeY = tensorY->GetDataType();

    // x / y数据类型必须一致
    if (dtypeX != dtypeY) {
        return ge::GRAPH_FAILED;
    }

    // 题目只支持float32 / float16
    if (dtypeX != ge::DT_FLOAT &&
        dtypeX != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }

    // 输入总元素数量
    uint32_t lengthX =
        static_cast<uint32_t>(tensorX->GetShapeSize());

    uint32_t lengthY =
        static_cast<uint32_t>(tensorY->GetShapeSize());

    if (lengthX == 0 || lengthX != lengthY) {
        return ge::GRAPH_FAILED;
    }

    // ================================================================
    // 3. 配置TilingKey
    //
    // tiling_key_mul.h中：
    //
    // DT_X:
    //     C_DT_FLOAT16
    //     C_DT_FLOAT
    //
    // Kernel侧会自动生成：
    //
    // KernelMul<half>
    // KernelMul<float>
    //
    // ================================================================
    uint32_t DT_X = static_cast<uint32_t>(dtypeX);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // ================================================================
    // 4. 选择启动Core数量
    //
    // 本题shape固定：
    //
    //     (8, 2048)
    //
    // 总元素：
    //
    //     8 * 2048 = 16384
    //
    // 最优且简单的切法：
    //
    //     8个Core
    //     每Core处理2048个元素
    //
    // 如果实际机器可用Vector Core不足8个，
    // 则退化到4 / 2 / 1核。
    // ================================================================
    uint32_t blockDim = 1;

    if (numCoresAiv >= 8) {
        blockDim = 8;
    } else if (numCoresAiv >= 4) {
        blockDim = 4;
    } else if (numCoresAiv >= 2) {
        blockDim = 2;
    }

    /*
     * 保证length能够被blockDim整除。
     *
     * 对题目规定的16384元素，
     * 8/4/2/1全部都可以整除。
     */
    while (blockDim > 1 &&
           (lengthX % blockDim) != 0) {
        blockDim >>= 1;
    }

    uint32_t blockLength =
        lengthX / blockDim;

    // ================================================================
    // 5. Tile切分
    //
    // 使用Double Buffer：
    //
    // BUFFER_NUM = 2
    //
    // blockDim = 8时：
    //
    // blockLength = 2048
    //
    // tileNum = 8
    //
    // tileLength =
    //
    // 2048 / (8 * 2)
    // = 128
    //
    // Process总循环：
    //
    // 8 * 2 = 16次
    //
    // ================================================================
    uint32_t tileNum = DEFAULT_TILE_NUM;

    /*
     * 保证blockLength能够被
     *
     * tileNum * BUFFER_NUM
     *
     * 整除。
     */
    while (tileNum > 1 &&
           (blockLength % (tileNum * BUFFER_NUM)) != 0) {
        --tileNum;
    }

    uint32_t tileLength =
        blockLength /
        (tileNum * BUFFER_NUM);

    /*
     * DataCopy的GM<->UB数据搬运最好满足32B对齐。
     *
     * float32:
     *   8个元素 = 32Byte
     *
     * float16:
     *   16个元素 = 32Byte
     */
    uint32_t alignElements =
        (dtypeX == ge::DT_FLOAT)
            ? 8U
            : 16U;

    /*
     * 如果当前tileLength没有满足32Byte对齐，
     * 继续减小tileNum。
     *
     * 对本题固定shape，
     * tileLength=128，
     * FP16/FP32均满足要求。
     */
    while (tileNum > 1 &&
           (tileLength % alignElements) != 0) {
        --tileNum;

        if ((blockLength %
             (tileNum * BUFFER_NUM)) != 0) {
            continue;
        }

        tileLength =
            blockLength /
            (tileNum * BUFFER_NUM);
    }

    if ((blockLength %
         (tileNum * BUFFER_NUM)) != 0) {
        return ge::GRAPH_FAILED;
    }

    tileLength =
        blockLength /
        (tileNum * BUFFER_NUM);

    if ((tileLength % alignElements) != 0) {
        return ge::GRAPH_FAILED;
    }

    // ================================================================
    // 6. 填写TilingData
    // ================================================================
    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }

    tiling->length = lengthX;
    tiling->blockLength = blockLength;
    tiling->tileNum = tileNum;
    tiling->tileLength = tileLength;

    // ================================================================
    // 7. 配置Kernel启动核数
    // ================================================================
    context->SetBlockDim(blockDim);

    // ================================================================
    // 8. 本算子无需workspace
    // ================================================================
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    if (currentWorkspace != nullptr) {
        currentWorkspace[0] = 0;
    }

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


// ====================================================================
// Shape / DataType推导
// ====================================================================

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    /*
     * z = x * y
     *
     * 本题x/y shape完全一致，
     * 所以输出z的shape直接跟随x。
     */
    const gert::Shape *inputShape =
        context->GetInputShape(0);

    gert::Shape *outputShape =
        context->GetOutputShape(0);

    if (inputShape == nullptr ||
        outputShape == nullptr) {
        return GRAPH_FAILED;
    }

    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}


static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    /*
     * 输出dtype与输入x一致：
     *
     * float32 -> float32
     * float16 -> float16
     */
    const auto inputDataType =
        context->GetInputDataType(0);

    return context->SetOutputDataType(
        0,
        inputDataType);
}

}  // namespace ge


// ====================================================================
// 算子原型定义
// ====================================================================

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
