// Host侧Tiling实现

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mul_tiling.h"
#include "../op_kernel/tiling_key_mul.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    // ============================================================
    // 1. 获取平台信息
    // ============================================================

    auto platform =
        platform_ascendc::PlatformAscendC(
            context->GetPlatformInfo());

    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    // ============================================================
    // 2. 获取输入Tensor信息
    // ============================================================

    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);

    if (tensor_x == nullptr || tensor_y == nullptr) {
        return ge::GRAPH_PARAM_INVALID;
    }

    // 获取数据类型
    ge::DataType dtype_x = tensor_x->GetDataType();
    ge::DataType dtype_y = tensor_y->GetDataType();

    // Mul要求两个输入数据类型一致
    if (dtype_x != dtype_y) {
        return ge::GRAPH_PARAM_INVALID;
    }

    // ============================================================
    // 3. 获取输入数据长度
    // ============================================================

    uint32_t length_x =
        static_cast<uint32_t>(tensor_x->GetShapeSize());

    uint32_t length_y =
        static_cast<uint32_t>(tensor_y->GetShapeSize());

    // 两个输入的元素数量必须一致
    if (length_x == 0 ||
        length_x != length_y ||
        num_cores_aiv <= 0) {
        return ge::GRAPH_PARAM_INVALID;
    }

    // ============================================================
    // 4. 设置Tiling Key
    // ============================================================

    uint32_t DT_X =
        static_cast<uint32_t>(dtype_x);

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // ============================================================
    // 5. 计算Block数量
    // ============================================================

    /*
     * 为了避免：
     *
     *     length_x / block_num
     *
     * 出现无法整除的问题，
     * 这里选择不超过AIV核数的最大因子。
     *
     * 对本题：
     *
     *     shape = (8, 2048)
     *     length = 8 * 2048 = 16384
     *
     * 如果设备有48个AIV核：
     *
     *     16384 % 48 != 0
     *
     * 因此最终会选择：
     *
     *     block_num = 32
     *
     * 因为：
     *
     *     16384 / 32 = 512
     */

    uint32_t block_num =
        static_cast<uint32_t>(num_cores_aiv);

    if (block_num > length_x) {
        block_num = length_x;
    }

    while (block_num > 1 &&
           length_x % block_num != 0) {
        --block_num;
    }

    // 每一个Block负责的数据量
    uint32_t block_length =
        length_x / block_num;

    // ============================================================
    // 6. 计算tile数量
    // ============================================================

    /*
     * 使用双缓冲：
     *
     * BUFFER_NUM = 2
     *
     * 每个Block内部再划分成2个tile。
     *
     * 因此：
     *
     * tileLength =
     *     blockLength / tileNum / BUFFER_NUM
     *
     * 对本题：
     *
     * blockLength = 512
     * tileNum     = 2
     * BUFFER_NUM  = 2
     *
     * tileLength  = 128
     */

    uint32_t tile_num =
        (block_length % 4 == 0) ? 2 : 1;

    // ============================================================
    // 7. 填写Tiling结构体
    // ============================================================

    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->length = length_x;
    tiling->tileNum = tile_num;

    // ============================================================
    // 8. 设置Block数量
    // ============================================================

    context->SetBlockDim(block_num);

    // ============================================================
    // 9. 设置Workspace
    // ============================================================

    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


// ================================================================
// Shape / DataType 推导
// ================================================================

namespace ge {

static graphStatus InferShape(
    gert::InferShapeContext *context)
{
    const gert::Shape *input_shape =
        context->GetInputShape(0);

    gert::Shape *output_shape =
        context->GetOutputShape(0);

    if (input_shape == nullptr ||
        output_shape == nullptr) {
        return GRAPH_PARAM_INVALID;
    }

    // Mul是逐元素运算
    // 输出shape与输入x完全一致
    *output_shape = *input_shape;

    return GRAPH_SUCCESS;
}


static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    // 输出数据类型与输入x一致
    context->SetOutputDataType(
        0,
        context->GetInputDataType(0));

    return ge::GRAPH_SUCCESS;
}

}  // namespace ge


// ================================================================
// 算子原型定义
// ================================================================

namespace ops {

class Mul : public OpDef {

public:

    explicit Mul(const char *name)
        : OpDef(name)
    {
        // --------------------------------------------------------
        // 输入 x
        // --------------------------------------------------------

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

        // --------------------------------------------------------
        // 输入 y
        // --------------------------------------------------------

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

        // --------------------------------------------------------
        // 输出 z
        // --------------------------------------------------------

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

        // --------------------------------------------------------
        // Shape / DataType推导
        // --------------------------------------------------------

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        // --------------------------------------------------------
        // AICore配置
        // --------------------------------------------------------

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops