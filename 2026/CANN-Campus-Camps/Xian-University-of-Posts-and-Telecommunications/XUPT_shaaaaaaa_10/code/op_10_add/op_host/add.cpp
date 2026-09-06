// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/add_tiling.h"
#include "../op_kernel/tiling_key_add.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext *context) {
    // 获取平台信息
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t num_cores_aiv = platform.GetCoreNumAiv();

    // 获取输入信息
    const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
    ge::DataType dtype_x = tensor_x->GetDataType();
    uint32_t length_x = tensor_x->GetShapeSize();  // 8 * 2048 = 16384

    // 配置tiling key，区分 float / float16 两套kernel模板
    uint32_t DT_X = static_cast<uint32_t>(dtype_x);
    ASCENDC_TPL_SEL_PARAM(context, DT_X);

    // ---------- 计算tiling方案 ----------
    // 总长度是2的幂（16384），让核数从大到小找到能整除总长度的值，
    // 保证每个核分到的数据量相同且按32元素对齐（fp16下为64B，满足DataCopy对齐要求）
    uint32_t blockDim = num_cores_aiv;
    if (blockDim == 0) blockDim = 1;
    while (blockDim > 1 && length_x % blockDim != 0) {
        blockDim--;
    }

    AddTilingData *tiling = context->GetTilingData<AddTilingData>();
    tiling->length  = length_x;             // 总长度
    tiling->perCore = length_x / blockDim;  // 每核处理的元素个数

    // 配置启动核数
    context->SetBlockDim(blockDim);

    // 配置workspace大小
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
// 输出shape与输入x一致
static graphStatus InferShape(gert::InferShapeContext *context) {
    const gert::Shape *xShape = context->GetInputShape(0);
    gert::Shape *zShape = context->GetOutputShape(0);
    if (xShape == nullptr || zShape == nullptr) {
        return GRAPH_FAILED;
    }
    *zShape = *xShape;
    return GRAPH_SUCCESS;
}

// 输出dtype与输入一致
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    ge::DataType dt = context->GetInputDataType(0);
    context->SetOutputDataType(0, dt);
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class Add : public OpDef {
public:
    explicit Add(const char *name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(Add);
}  // namespace ops

