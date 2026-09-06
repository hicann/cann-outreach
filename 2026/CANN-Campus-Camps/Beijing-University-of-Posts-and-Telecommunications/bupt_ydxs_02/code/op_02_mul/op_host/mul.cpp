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
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    // 获取AIV Core数量
    int32_t num_cores_aiv = platform.GetCoreNumAiv();

    // 获取UB大小
    uint64_t ub_size = 0;
    platform.GetCoreMemSize(
        platform_ascendc::CoreMemType::UB,
        ub_size);

    // ============================================================
    // 2. 获取输入Tensor信息
    // ============================================================

    const gert::Tensor *tensor_x =
        context->GetRequiredInputTensor(0);

    const gert::Tensor *tensor_y =
        context->GetRequiredInputTensor(1);

    // 获取x的数据类型
    ge::DataType dtype_x = tensor_x->GetDataType();

    // 获取数据类型字节数
    int32_t dtype_size_x =
        ge::GetSizeByDataType(dtype_x);

    // 获取输入元素总数量
    uint32_t length_x =
        tensor_x->GetShapeSize();

    // y与x形状、dtype由算子约束保证一致
    (void)tensor_y;

    // ============================================================
    // 3. 配置Tiling Key
    // ============================================================

    uint32_t DT_X =
        static_cast<uint32_t>(dtype_x);

    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // ============================================================
    // 4. 计算实际使用的Core数量
    // ============================================================

    uint32_t core_num =
        static_cast<uint32_t>(num_cores_aiv);

    // Core数量不能超过数据元素数量
    if (length_x < core_num) {
        core_num = length_x;
    }

    // 防止Core数量为0
    if (core_num == 0) {
        core_num = 1;
    }

    // ============================================================
    // 5. 计算每个Core处理的数据量
    // ============================================================

    // 采用向上取整
    uint32_t block_length =
        (length_x + core_num - 1) / core_num;

    // ============================================================
    // 6. 根据UB大小计算Tile大小
    // ============================================================

    // 一个Tile同时需要：
    //
    // xLocal : tileLength * dtype_size
    // yLocal : tileLength * dtype_size
    // zLocal : tileLength * dtype_size
    //
    // 因此需要3份Buffer

    uint64_t element_num_by_ub =
        ub_size / (3 * static_cast<uint64_t>(dtype_size_x));

    // AscendC Vector数据通常需要32字节对齐
    uint32_t align_num =
        32 / static_cast<uint32_t>(dtype_size_x);

    uint32_t tile_length =
        static_cast<uint32_t>(element_num_by_ub);

    // 向下对齐到32字节
    if (tile_length >= align_num) {
        tile_length =
            (tile_length / align_num) * align_num;
    }

    // 防止Tile长度为0
    if (tile_length == 0) {
        tile_length = align_num;
    }

    // 当前Core的数据量小于Tile时
    if (block_length < tile_length) {

        // 将实际数据量向上对齐到32字节
        uint32_t aligned_block_length =
            ((block_length * dtype_size_x + 31) / 32) *
            (32 / dtype_size_x);

        tile_length = aligned_block_length;
    }

    // ============================================================
    // 7. 计算Tile数量
    // ============================================================

    uint32_t tile_num = 0;
    uint32_t last_tile_length = 0;

    if (block_length > 0) {
        tile_num =
            block_length / tile_length;

        last_tile_length =
            block_length % tile_length;

        // 存在尾块
        if (last_tile_length != 0) {
            tile_num += 1;
        }
    }

    // ============================================================
    // 8. 写入TilingData
    // ============================================================

    MulTilingData *tiling =
        context->GetTilingData<MulTilingData>();

    tiling->length = length_x;
    tiling->blockLength = block_length;
    tiling->tileLength = tile_length;
    tiling->tileNum = tile_num;
    tiling->lastTileLength = last_tile_length;

    // ============================================================
    // 9. 设置启动Core数量
    // ============================================================

    context->SetBlockDim(core_num);

    // ============================================================
    // 10. 设置Workspace
    // ============================================================

    size_t *currentWorkspace =
        context->GetWorkspaceSizes(1);

    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling


// ================================================================
// InferShape
// ================================================================

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    // 获取输入x的Shape
    const gert::Shape *x_shape =
        context->GetInputShape(0);

    // 获取输出z的Shape
    gert::Shape *z_shape =
        context->GetOutputShape(0);

    // z = x * y
    //
    // 输出Shape与x一致
    *z_shape = *x_shape;

    return GRAPH_SUCCESS;
}


// ================================================================
// InferDataType
// ================================================================

static graphStatus InferDataType(
    gert::InferDataTypeContext *context)
{
    // 获取输入x的数据类型
    ge::DataType x_dtype =
        context->GetInputDataType(0);

    // 输出z的数据类型与x一致
    context->SetOutputDataType(0, x_dtype);

    return GRAPH_SUCCESS;
}

}  // namespace ge


// ================================================================
// 算子定义
// ================================================================

namespace ops {

class Mul : public OpDef {
public:

    explicit Mul(const char *name)
        : OpDef(name)
    {
        // --------------------------------------------------------
        // 输入x
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
        // 输入y
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
        // 输出z
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
        // 注册Shape和DataType推导
        // --------------------------------------------------------

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        // --------------------------------------------------------
        // 注册Tiling
        // --------------------------------------------------------

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Mul);

}  // namespace ops