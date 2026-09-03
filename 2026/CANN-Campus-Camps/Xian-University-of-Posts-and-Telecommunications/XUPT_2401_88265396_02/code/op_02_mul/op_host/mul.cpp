// Host侧Tiling实现
#include "register/op_def_registry.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"


namespace optiling {

static ge::graphStatus TilingFunc(
    gert::TilingContext *context)
{
    // =============================================
    // 获取输入
    // =============================================

    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);


    // =============================================
    // 获取总元素数量
    // =============================================

    uint32_t length_x =
        tensor_x->GetShapeSize();


    // =============================================
    // 获取数据类型
    // =============================================

    ge::DataType dtype_x =
        tensor_x->GetDataType();

    ge::DataType dtype_y =
        tensor_y->GetDataType();


    uint32_t DT_X =
        static_cast<uint32_t>(
            dtype_x);

    uint32_t DT_Y =
        static_cast<uint32_t>(
            dtype_y);

    /*
     * 题目规定：
     *
     * z.dtype = x.dtype
     */
    uint32_t DT_Z =
        static_cast<uint32_t>(
            dtype_x);


    // =============================================
    // 配置模板参数
    // =============================================

    ASCENDC_TPL_SEL_PARAM(
        context,
        DT_X,
        DT_Y,
        DT_Z);


    // =============================================
    // 填充TilingData
    // =============================================

    MulTilingData *tiling =
        context->GetTilingData<
            MulTilingData>();

    tiling->length =
        length_x;


    // =============================================
    // 正确性优先：
    //
    // 使用单Block。
    //
    // Kernel内部按Tile遍历整个Tensor，
    // 因此不存在：
    //
    // length % coreNum
    //
    // 导致的尾部数据丢失问题。
    // =============================================

    context->SetBlockDim(1);


    // =============================================
    // Workspace
    // =============================================

    size_t *workspaceSize =
        context->GetWorkspaceSizes(1);

    workspaceSize[0] = 0;


    return ge::GRAPH_SUCCESS;
}

} // namespace optiling



namespace ge {

// ============================================================
// Shape推导
// ============================================================

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    /*
     * element-wise Mul:
     *
     * z.shape = x.shape
     */

    const gert::Shape *xShape =
        context->GetInputShape(0);

    gert::Shape *zShape =
        context->GetOutputShape(0);

    *zShape =
        *xShape;

    return GRAPH_SUCCESS;
}


// ============================================================
// DType推导
// ============================================================

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    /*
     * z.dtype = x.dtype
     */

    ge::DataType xType =
        context->GetInputDataType(0);

    context->SetOutputDataType(
        0,
        xType);

    return GRAPH_SUCCESS;
}

} // namespace ge



namespace ops {

class Mul : public OpDef {
public:

    explicit Mul(
        const char *name)
        : OpDef(name)
    {
        // ====================================================
        // x
        // ====================================================

        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });


        // ====================================================
        // y
        // ====================================================

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });


        // ====================================================
        // z
        // ====================================================

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({
                ge::DT_FLOAT16,
                ge::DT_FLOAT
            })
            .Format({
                ge::FORMAT_ND,
                ge::FORMAT_ND
            });


        // ====================================================
        // 推导函数
        // ====================================================

        this->SetInferShape(
                ge::InferShape)
            .SetInferDataType(
                ge::InferDataType);


        // ====================================================
        // AI Core
        // ====================================================

        this->AICore()
            .SetTiling(
                optiling::TilingFunc)
            .AddConfig(
                "ascend910b");
    }
};


OP_ADD(Mul);

} // namespace ops