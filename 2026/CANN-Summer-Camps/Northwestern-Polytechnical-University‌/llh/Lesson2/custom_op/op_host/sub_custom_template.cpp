/**
 * @file    sub_custom_template.cpp
 * @brief   SubCustomTemplate 算子 — Host 侧算子注册与自适应 Tiling
 *
 * ============================================================================
 * Host 侧三步
 * ============================================================================
 *
 *   Step A  TilingFunc()  自适应计算分片参数 (totalLength + tileNum)
 *   Step B  InferShape / InferDataType  输出 shape/dtype 自动推导
 *   Step C  OpDef 注册    声明算子接口 (input/output/dtype/format/后端)
 *
 * ============================================================================
 * 自适应 Tiling 策略
 * ============================================================================
 *
 *   与 Lesson3 固定 tileNum=45 不同，本算子采用自适应策略:
 *
 *   1. 目标 tileLength = 256 元素 (平衡 MTE 带宽效率与 UB 占用)
 *   2. BlockDim = min(ceil(totalLength / tileLength), 40)
 *      → 数据大时用满 40 核，数据小时自动减少核数，避免单核数据过少
 *   3. tileNum = blockLength / (tileLength × BUFFER_NUM)
 *      → 确保 tileLength 接近目标值
 *
 *   示例 (FP32, totalLength=16384):
 *     BlockDim  = min(16384/256, 40) = 32 核
 *     blockLength = 16384/32 = 512
 *     tileNum  = 512/(256×2) = 1
 *     tileLength = 512/1/2 = 256 ✓
 *
 * ============================================================================
 * 数据流
 * ============================================================================
 *
 *   Host (CPU)                           Device (NPU)
 *   ──────────                          ────────────
 *   TilingFunc()
 *     ├─ 读取 input shape → totalLength
 *     ├─ 自适应计算 BlockDim
 *     └─ 打包 SubCustomTemplateTilingData
 *              ↓
 *   框架自动拷贝到 GM(tiling buffer)
 *              ↓
 *                                  Kernel 启动, BlockDim 核并行
 *                                  每核通过 GET_TILING_DATA_WITH_STRUCT
 *                                  读取 totalLength + tileNum
 *                                  执行 CopyIn→Sub→CopyOut 流水线
 */

#include "../op_kernel/sub_custom_template_tiling.h"
#include "register/op_def_registry.h"

// 目标 tile 大小 (元素数) — MTE 长 burst 效率与 UB 占用的平衡点
constexpr uint32_t TARGET_TILE_LENGTH = 256;

// =============================================================================
// Part A — 自适应 Tiling 函数
// =============================================================================
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SubCustomTemplateTilingData *tiling =
        context->GetTilingData<SubCustomTemplateTilingData>();

    uint32_t totalLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();

    // --- 自适应 BlockDim ---
    // 每核至少处理 TARGET_TILE_LENGTH 个元素, 最多用 40 核
    uint32_t blockDim = (totalLength + TARGET_TILE_LENGTH - 1) / TARGET_TILE_LENGTH;
    if (blockDim > 40) blockDim = 40;
    if (blockDim < 1)  blockDim = 1;

    // 确保 BlockDim 能整除 totalLength, 避免尾块数据丢失
    // 例: totalLength=16384 → blockDim 从 40 回退到 32 (16384%32==0)
    while (blockDim > 1 && totalLength % blockDim != 0) {
        blockDim--;
    }
    context->SetBlockDim(blockDim);

    // --- 计算 tileNum ---
    // blockLength = totalLength / blockDim
    // tileLength ≈ TARGET_TILE_LENGTH
    // tileNum    = blockLength / (TARGET_TILE_LENGTH * BUFFER_NUM)
    uint32_t blockLength = totalLength / blockDim;
    uint32_t tileNum = blockLength / (TARGET_TILE_LENGTH * 2);  // BUFFER_NUM=2
    if (tileNum < 1) tileNum = 1;  // blockLength < 512 时至少切 1 轮

    // --- 打包参数 ---
    tiling->totalLength = totalLength;
    tiling->tileNum     = tileNum;

    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

// =============================================================================
// Part B — Shape / DataType 自动推导
// =============================================================================
// Element-wise Sub: 输出 shape == 输入 shape, 输出 dtype == 输入 dtype
//
namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

// =============================================================================
// Part C — 算子注册
// =============================================================================
// 向 CANN 框架注册 SubCustomTemplate 算子:
//   - 算子类名 SubCustomTemplate → OP_ADD 宏生成 aclnnSubCustomTemplate API
//   - 支持 float16 / float32 两种精度
//   - 绑定 sopra 的 Tiling / InferShape / InferDataType 函数
//   - 目标芯片 ascend910b
//
namespace ops {
class SubCustomTemplate : public OpDef {
public:
    explicit SubCustomTemplate(const char* name) : OpDef(name)
    {
        // ---- 输入 x ----
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // ---- 输入 y ----
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // ---- 输出 z = x - y ----
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // 绑定推导函数
        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        // 绑定 Tiling 函数 + 目标芯片
        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

// 注册: 类名 SubCustomTemplate → 外部 API 名 aclnnSubCustomTemplate
OP_ADD(SubCustomTemplate);
}  // namespace ops
