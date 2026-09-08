// Host侧Tiling实现

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"


// ============================================================
// Tiling
// ============================================================

namespace optiling {

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    // --------------------------------------------------------
    // 1. 获取 NPU 平台信息
    // --------------------------------------------------------
    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    int32_t num_cores_aiv =
        platform.GetCoreNumAiv();

    uint64_t ub_size = 0;

    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ub_size);


    // --------------------------------------------------------
    // 2. 获取两个输入 Tensor
    // --------------------------------------------------------
    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);


    // --------------------------------------------------------
    // 3. 获取输入数据类型
    // --------------------------------------------------------
    ge::DataType dtype_x =
        tensor_x->GetDataType();

    int dtype_size_x =
        ge::GetSizeByDataType(dtype_x);


    // --------------------------------------------------------
    // 4. 获取 Tensor 元素数量
    //
    // 本题：
    // shape = (8, 2048)
    //
    // length = 8 * 2048 = 16384
    // --------------------------------------------------------
    uint32_t length_x =
        tensor_x->GetShapeSize();

    uint32_t size_x =
        tensor_x->GetSize();


    // --------------------------------------------------------
    // 5. 配置 TilingKey
    //
    // 根据 x 的数据类型区分：
    // float16 / float32
    // --------------------------------------------------------
    uint32_t DT_X =
        static_cast<uint32_t>(dtype_x);

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_X);


    // --------------------------------------------------------
    // 6. 填写 Tiling 数据
    // --------------------------------------------------------
    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->length =
        length_x;


    // --------------------------------------------------------
    // 7. 配置 Kernel 启动核数
    //
    // 输入为：
    //
    // (8, 2048)
    //
    // 总元素：
    // 16384
    //
    // 固定使用 8 个 Vector Core：
    //
    // 每核：
    // 16384 / 8 = 2048
    //
    // 可以整除，不存在尾块问题。
    // --------------------------------------------------------
    uint32_t blockDim = 8;

    if (num_cores_aiv < 8) {
        if (num_cores_aiv >= 4) {
            blockDim = 4;
        } else if (num_cores_aiv >= 2) {
            blockDim = 2;
        } else {
            blockDim = 1;
        }
    }

    context->SetBlockDim(
        blockDim);


    // --------------------------------------------------------
    // 8. Workspace
    //
    // 本题不需要额外 Workspace
    // --------------------------------------------------------
    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;


    // --------------------------------------------------------
    // 避免模板中的辅助变量产生未使用警告
    // --------------------------------------------------------
    (void)tensor_y;
    (void)dtype_size_x;
    (void)size_x;
    (void)ub_size;


    return ge::GRAPH_SUCCESS;
}

} // namespace optiling



// ============================================================
// Shape / DataType 推导
// ============================================================

namespace ge {


// ------------------------------------------------------------
// InferShape
//
// 输出 z 的 shape 与输入 x 一致
//
// x = (8, 2048)
//
// 因此：
//
// z = (8, 2048)
// ------------------------------------------------------------
static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const gert::Shape *inputShape =
        context->GetInputShape(0);

    gert::Shape *outputShape =
        context->GetOutputShape(0);

    *outputShape =
        *inputShape;

    return GRAPH_SUCCESS;
}


// ------------------------------------------------------------
// InferDataType
//
// z 的数据类型跟 x 一致
//
// x=float32 -> z=float32
// x=float16 -> z=float16
// ------------------------------------------------------------
static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    const auto inputDataType =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        inputDataType);

    return ge::GRAPH_SUCCESS;
}

} // namespace ge



// ============================================================
// 算子原型注册
// ============================================================

namespace ops {

class Mul : public OpDef {

public:

    explicit Mul(
        const char *name)
        : OpDef(name)
    {
        // ----------------------------------------------------
        // 输入 x
        // ----------------------------------------------------
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


        // ----------------------------------------------------
        // 输入 y
        // ----------------------------------------------------
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


        // ----------------------------------------------------
        // 输出 z
        // ----------------------------------------------------
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


        // ----------------------------------------------------
        // Shape + DataType 推导
        // ----------------------------------------------------
        this->SetInferShape(
                ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);


        // ----------------------------------------------------
        // AI Core 配置
        // ----------------------------------------------------
        this->AICore()
            .SetTiling(
                optiling::TilingFunc)
            .AddConfig(
                "ascend910b");
    }
};


// 注册 Mul 算子
OP_ADD(Mul);

} // namespace ops