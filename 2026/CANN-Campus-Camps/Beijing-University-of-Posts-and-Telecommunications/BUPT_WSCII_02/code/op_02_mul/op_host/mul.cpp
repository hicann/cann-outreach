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

    uint64_t ubSize;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ubSize
    );

    // 获取输入 Tensor
    const gert::Tensor *tensorX =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensorY =
        context->GetRequiredInputTensor(1);

    // 获取输入数据类型
    ge::DataType dtypeX = tensorX->GetDataType();

    int dtypeSizeX =
        ge::GetSizeByDataType(dtypeX);

    // 总元素个数
    uint32_t lengthX =
        static_cast<uint32_t>(tensorX->GetShapeSize());

    // 设置 Tiling Key
    uint32_t DT_X =
        static_cast<uint32_t>(dtypeX);

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // 填充 TilingData
    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->length = lengthX;

    /*
     * BUFFER_NUM = 2
     *
     * 每核：
     * 16384 / 8 = 2048 elements
     *
     * tileNum = 1 时：
     * tileLength = 2048 / 1 / 2 = 1024
     *
     * Process 共执行 2 次。
     */
    tiling->tileNum = 1;

    /*
     * 本题输入固定为 (8, 2048)
     *
     * 使用 8 个核：
     * 每核恰好处理 2048 个元素。
     *
     * 不建议直接 SetBlockDim(numCoresAiv)，
     * 因为 numCoresAiv 不一定整除 16384。
     */
    constexpr uint32_t BLOCK_DIM = 8;
    context->SetBlockDim(BLOCK_DIM);

    // 无需 workspace
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

    *zShape = *xShape;

    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    const auto inputDataType =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        inputDataType
    );

    return ge::GRAPH_SUCCESS;
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