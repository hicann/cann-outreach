/*!
 * \file gelu.cpp
 * \brief GELU 算子 Host Tiling 实现
 */

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/gelu_tiling.h"
#include "../op_kernel/tiling_key_gelu.h"

namespace optiling {

// ============================================================
// 常量定义
// ============================================================

// 一个 DataBlock 为 32 Bytes
static constexpr uint32_t GELU_DATA_BLOCK_BYTES = 32;

// 希望每个 Core 至少处理大约 256 个元素
static constexpr uint32_t GELU_ELEMENTS_PER_CORE = 256;

// UB 每次处理的元素数量
static constexpr uint32_t GELU_TILE_LENGTH = 2048;


// ============================================================
// 向上取整
// ============================================================

static inline uint32_t CeilDiv(
    uint32_t a,
    uint32_t b)
{
    return (a + b - 1U) / b;
}


// ============================================================
// 向上对齐
// ============================================================

static inline uint32_t AlignUp(
    uint32_t value,
    uint32_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}


// ============================================================
// Tiling 函数
// ============================================================

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    // --------------------------------------------------------
    // 1. 获取平台信息
    // --------------------------------------------------------

    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    uint32_t numCores =
        static_cast<uint32_t>(
            platform.GetCoreNumAiv());

    if (numCores == 0U) {
        numCores = 1U;
    }


    // --------------------------------------------------------
    // 2. 获取输入 Tensor
    // --------------------------------------------------------

    const gert::Tensor *tensor_input_x =
        context->GetRequiredInputTensor(0);

    if (tensor_input_x == nullptr) {
        return ge::GRAPH_FAILED;
    }


    // --------------------------------------------------------
    // 3. 获取数据类型
    // --------------------------------------------------------

    ge::DataType dtype_input_x =
        tensor_input_x->GetDataType();


    // --------------------------------------------------------
    // 4. 检查数据类型
    //
    // 题目只允许：
    // float16
    // float32
    // --------------------------------------------------------

    if (dtype_input_x != ge::DT_FLOAT16 &&
        dtype_input_x != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }


    // --------------------------------------------------------
    // 5. 获取元素数量
    //
    // 不管输入是：
    //
    // [N]
    // [B, N]
    // [B, C, N]
    // [B, C, H, N]
    //
    // GetShapeSize() 都返回总元素数量。
    // --------------------------------------------------------

    const uint32_t totalLength =
        static_cast<uint32_t>(
            tensor_input_x->GetShapeSize());


    // --------------------------------------------------------
    // 6. 设置 Tiling Key
    //
    // 这里必须使用你原来的 DT_INPUT_X 模板参数。
    //
    // float32 -> C_DT_FLOAT
    // float16 -> C_DT_FLOAT16
    // --------------------------------------------------------

    uint32_t DT_INPUT_X =
        static_cast<uint32_t>(dtype_input_x);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_INPUT_X);


    // --------------------------------------------------------
    // 7. 获取 TilingData
    // --------------------------------------------------------

    GeluTilingData *tiling =
        context->GetTilingData<GeluTilingData>();


    // --------------------------------------------------------
    // 8. 保存总元素数量
    // --------------------------------------------------------

    tiling->totalLength =
        totalLength;


    // --------------------------------------------------------
    // 9. 空输入保护
    //
    // 题目规定 N >= 1，正常不会出现。
    // --------------------------------------------------------

    if (totalLength == 0U) {

        tiling->blockLength = 0U;
        tiling->lastBlockLength = 0U;
        tiling->tileLength = GELU_TILE_LENGTH;

        context->SetBlockDim(1);

        size_t *currentWorkspace =
            context->GetWorkspaceSizes(1);

        currentWorkspace[0] = 0;

        return ge::GRAPH_SUCCESS;
    }


    // --------------------------------------------------------
    // 10. 获取单个元素字节数
    // --------------------------------------------------------

    const uint32_t elementSize =
        static_cast<uint32_t>(
            ge::GetSizeByDataType(dtype_input_x));


    // --------------------------------------------------------
    // 11. 计算 32 Byte 对应的元素数量
    //
    // float16:
    //
    // 32 / 2 = 16
    //
    // float32:
    //
    // 32 / 4 = 8
    // --------------------------------------------------------

    const uint32_t alignElements =
        GELU_DATA_BLOCK_BYTES / elementSize;


    // --------------------------------------------------------
    // 12. 根据输入长度计算使用 Core 数
    //
    // 例如：
    //
    // N = 10240
    // 每 Core 目标约 256
    //
    // 10240 / 256 = 40
    //
    // 如果平台有 40 个 Core：
    //
    // 使用 40 Core
    //
    // N 很小时，不会强行使用全部 Core。
    // --------------------------------------------------------

    uint32_t usedCoreNum =
        CeilDiv(
            totalLength,
            GELU_ELEMENTS_PER_CORE);

    if (usedCoreNum == 0U) {
        usedCoreNum = 1U;
    }

    if (usedCoreNum > numCores) {
        usedCoreNum = numCores;
    }


    // --------------------------------------------------------
    // 13. 计算每个普通 Core 的数据长度
    //
    // 先平均分配，再向 32 Byte 对齐。
    // --------------------------------------------------------

    uint32_t blockLength =
        CeilDiv(
            totalLength,
            usedCoreNum);

    blockLength =
        AlignUp(
            blockLength,
            alignElements);


    // --------------------------------------------------------
    // 14. 防止对齐以后造成无效 Core
    //
    // 例如：
    //
    // totalLength = 1025
    //
    // blockLength 经过 32Byte 对齐后可能变大。
    //
    // 如果前面的 Core 已经覆盖全部数据，
    // 就减少 Core 数量。
    // --------------------------------------------------------

    while (usedCoreNum > 1U &&
           blockLength * (usedCoreNum - 1U)
               >= totalLength) {

        --usedCoreNum;

        blockLength =
            CeilDiv(
                totalLength,
                usedCoreNum);

        blockLength =
            AlignUp(
                blockLength,
                alignElements);
    }


    // --------------------------------------------------------
    // 15. 计算最后一个 Core 的真实长度
    // --------------------------------------------------------

    uint32_t lastBlockLength =
        totalLength -
        blockLength * (usedCoreNum - 1U);


    // --------------------------------------------------------
    // 16. 保存 Tiling 参数
    // --------------------------------------------------------

    tiling->blockLength =
        blockLength;

    tiling->lastBlockLength =
        lastBlockLength;

    tiling->tileLength =
        GELU_TILE_LENGTH;


    // --------------------------------------------------------
    // 17. 设置实际启动 Core 数
    // --------------------------------------------------------

    context->SetBlockDim(
        usedCoreNum);


    // --------------------------------------------------------
    // 18. GELU 不需要额外 Workspace
    // --------------------------------------------------------

    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;


    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


// ============================================================
// InferShape
// ============================================================

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    // GELU 是逐元素计算：
    //
    // output shape == input shape

    const gert::Shape *inputShape =
        context->GetInputShape(0);

    gert::Shape *outputShape =
        context->GetOutputShape(0);

    *outputShape = *inputShape;

    return GRAPH_SUCCESS;
}


// ============================================================
// InferDataType
// ============================================================

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    // GELU：
    //
    // output dtype == input dtype

    const ge::DataType inputDataType =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        inputDataType);

    return GRAPH_SUCCESS;
}

}  // namespace ge


// ============================================================
// GELU 算子定义
// ============================================================

namespace ops {

class Gelu : public OpDef {

public:

    explicit Gelu(const char *name)
        : OpDef(name)
    {
        // ----------------------------------------------------
        // Input
        // ----------------------------------------------------

        this->Input("input_x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });


        // ----------------------------------------------------
        // Output
        // ----------------------------------------------------

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });


        // ----------------------------------------------------
        // Shape / DataType 推导
        // ----------------------------------------------------

        this->SetInferShape(
                ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);


        // ----------------------------------------------------
        // AI Core
        // ----------------------------------------------------

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};


OP_ADD(Gelu);

}  // namespace ops